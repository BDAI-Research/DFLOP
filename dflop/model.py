import os
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from torchtune.modules import TransformerDecoder
from torchtune.modules.attention_utils import _MaskType
from torchtune.modules.loss import LinearCrossEntropyLoss
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX
)
from llava.model.llava_arch import unpad_image
from llava.mm_utils import get_anyres_image_grid_shape
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionModel
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig
from .internvit_modules import InternVisionConfig, InternVisionModel


sig_lip_configs = {
    "400m": SigLipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
    ),
    "1b": SigLipVisionConfig(
        hidden_size=1728,
        intermediate_size=6912,
        num_hidden_layers=27,
        num_attention_heads=32,
    ),
    "6b": SigLipVisionConfig(
        hidden_size=2304,
        intermediate_size=8608,
        num_hidden_layers=40,
        num_attention_heads=48,
    ),
}

internvit_configs = {
    "6b": InternVisionConfig(
        hidden_size=3200,
        intermediate_size=12800,
        num_hidden_layers=30,
        num_attention_heads=16,
        image_size=448,
        hidden_act="gelu",
        norm_type="rms_norm",
        num_channels=3,
        patch_size=14,
        qk_normalization=True,
        qkv_bias=False,
        torch_dtype="bfloat16",
        use_bfloat16=False,
        use_flash_attn=True,
    ),
    "1b": InternVisionConfig(
        hidden_size=1792,
        intermediate_size=7168,
        num_hidden_layers=28,
        num_attention_heads=28,
        image_size=448,
        hidden_act="gelu",
        norm_type="rms_norm",
        num_channels=3,
        patch_size=14,
        qk_normalization=True,
        qkv_bias=False,
        torch_dtype="float16",
        use_bfloat16=False,
        use_flash_attn=True,
    ),
    "300m": InternVisionConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        image_size=448,
        num_attention_heads=16,
        hidden_act="gelu",
        norm_type="layer_norm",
        num_channels=3,
        patch_size=14,
        qk_normalization=False,
        qkv_bias=True,
        torch_dtype="float16",
        use_bfloat16=False,
        use_flash_attn=True,
    ),
}

qwen_configs = {
    "7b": {
        "num_layers": 28,
        "num_heads": 28,
        "num_kv_heads": 4,
        "embed_dim": 3584,
        "intermediate_dim": 18944,
    },
    "32b": {
        "num_layers": 64,
        "num_heads": 40,
        "num_kv_heads": 8,
        "embed_dim": 5120,
        "intermediate_dim": 27648,
    },
    "72b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "embed_dim": 8192,
        "intermediate_dim": 29568,
    },
}

llama_configs = {
    "8b": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "embed_dim": 4096,
        "intermediate_dim": 14336,
    },
    "70b": {
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "embed_dim": 8192,
        "intermediate_dim": 28672,
    },
}

vision_configs = {"siglip": sig_lip_configs, "internvit": internvit_configs}
llm_configs = {"qwen2.5": qwen_configs, "llama3": llama_configs}


def pixel_shuffle(x, scale_factor=0.5):
    x = x.contiguous()
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.reshape(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x

class LLaVAOVMMConfig:
    def __init__(self, mm_projector_type, mm_hidden_size, hidden_size):
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size

class LLaVAOVSigLipModule(nn.Module):
    def __init__(self, vision_config, mm_config, model_dtype, device):
        super().__init__()
        vit_model = SigLipVisionModel(vision_config).vision_model
        vit_model = vit_model.to(model_dtype)
        vit_model = vit_model.to(device)
        
        self.encoder = vit_model.encoder
        self.embeddings = vit_model.embeddings
        self.mm_projector = build_vision_projector(mm_config).to(model_dtype)
        self.mm_projector = self.mm_projector.to(device)
        self.device = device
        self.dtype = model_dtype
        
    def forward(self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        assert isinstance(pixel_values, torch.Tensor) , f"images should be a Tensor, got {type(pixel_values)}"
        
        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        
        output = self.mm_projector(last_hidden_state)
        assert output.shape[-2] == 729
        return output

class LLaVAOVInternVitModule(nn.Module):
    def __init__(self, vision_config, mm_config, model_dtype, device):
        super().__init__()
        vit_model = InternVisionModel(vision_config)
        vit_model = vit_model.to(model_dtype)
        vit_model = vit_model.to(device)
        
        self.encoder = vit_model.encoder
        self.embeddings = vit_model.embeddings
        self.mm_projector = build_vision_projector(mm_config).to(model_dtype)
        self.mm_projector = self.mm_projector.to(device)
        self.device = device
        self.dtype = model_dtype
        
    def forward(self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        assert isinstance(pixel_values, torch.Tensor) , f"images should be a Tensor, got {type(pixel_values)}"
        
        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(hidden_states)
        last_hidden_state = encoder_outputs[0]
        
        output = self.mm_projector(last_hidden_state)
        return output

class InterVLSigLipModule(nn.Module):
    def __init__(self, vision_config, llm_config, model_dtype, device):
        super().__init__()
        vit_model = SigLipVisionModel(vision_config).vision_model
        vit_model = vit_model.to(model_dtype)
        vit_model = vit_model.to(device)
        self.downsample_ratio = 0.5
        self.encoder = vit_model.encoder
        self.embeddings = vit_model.embeddings
        vit_hidden_size = vision_config.hidden_size
        self.mm_projector = nn.Sequential(
                    nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
                    nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_config["embed_dim"]),
                    nn.GELU(),
                    nn.Linear(llm_config["embed_dim"], llm_config["embed_dim"])
                )

        self.mm_projector = self.mm_projector.to(device)
        self.device = device
        self.dtype = model_dtype
        
    def forward(self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        assert isinstance(pixel_values, torch.Tensor) , f"images should be a Tensor, got {type(pixel_values)}"
        
        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        vit_embeds = encoder_outputs[0]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = vit_embeds.contiguous()
        output = self.mm_projector(vit_embeds)
        assert output.shape[-2] == 729
        return output

class InternVLInternVitModule(nn.Module):
    def __init__(self, vision_config, llm_config, model_dtype, device):
        super().__init__()
        vit_model = InternVisionModel(vision_config)
        vit_model = vit_model.to(model_dtype)
        vit_model = vit_model.to(device)
        self.downsample_ratio = 0.5
        self.encoder = vit_model.encoder
        self.embeddings = vit_model.embeddings
        vit_hidden_size = vision_config.hidden_size
        self.mm_projector = nn.Sequential(
                            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
                            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_config["embed_dim"]),
                            nn.GELU(),
                            nn.Linear(llm_config["embed_dim"], llm_config["embed_dim"])
                        )

        self.mm_projector = self.mm_projector.to(device)
        self.device = device
        self.dtype = model_dtype
        
    def forward(self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # print(f"[Rank : {self.device.index}]Input : {pixel_values}")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        assert isinstance(pixel_values, torch.Tensor) , f"images should be a Tensor, got {type(pixel_values)}"
        
        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(hidden_states)
        vit_embeds = encoder_outputs[0]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = vit_embeds.contiguous()
        output = self.mm_projector(vit_embeds)
        return output

class LLaVAOVInternVitModule(nn.Module):
    def __init__(self, vision_config, mm_config, model_dtype, device):
        super().__init__()
        vit_model = InternVisionModel(vision_config)
        vit_model = vit_model.to(model_dtype)
        vit_model = vit_model.to(device)
        
        self.encoder = vit_model.encoder
        self.embeddings = vit_model.embeddings
        self.mm_projector = build_vision_projector(mm_config).to(model_dtype)
        self.mm_projector = self.mm_projector.to(device)
        self.device = device
        self.dtype = model_dtype
        
    def forward(self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        assert isinstance(pixel_values, torch.Tensor) , f"images should be a Tensor, got {type(pixel_values)}"
        
        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(hidden_states)
        last_hidden_state = encoder_outputs[0]
        
        output = self.mm_projector(last_hidden_state)
        return output

class VisionPipeStage(nn.Module):
    def __init__(self, vision_model, vision_stages, stage_id, device, model_dtype, is_internvl=False):
        super().__init__()
        self.downsample_ratio = 0.5
        num_layers = len(vision_model.encoder.layers)
        layers_per_stage = num_layers // len(vision_stages)
        remain_layers = num_layers % len(vision_stages)
        stage_layer_counts = [layers_per_stage + (1 if i < remain_layers else 0) for i in range(len(vision_stages))]
        print(f"stage_layer_counts : {stage_layer_counts}, num_layers : {num_layers}, len(vision_stages) : {len(vision_stages)}, remain_layers : {remain_layers}")
        stage_idx = vision_stages.index(stage_id)
        start_idx = sum(stage_layer_counts[:stage_idx])
        end_idx = start_idx + stage_layer_counts[stage_idx]

        self.encoder = vision_model.encoder
        self.encoder.layers = nn.ModuleList(vision_model.encoder.layers[start_idx:end_idx])
        self.embeddings = None
        self.mm_projector = None
        if stage_id == vision_stages[0]:
            self.embeddings = vision_model.embeddings
        if stage_id == vision_stages[-1]:
            self.mm_projector = vision_model.mm_projector
        self.device = device
        self.dtype = model_dtype
        self.is_internvl = is_internvl



    def forward(self, pixel_values):
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        if self.embeddings is not None:
            hidden_states = self.embeddings(pixel_values)
        else:
            hidden_states = pixel_values
        encoder_outputs = self.encoder(hidden_states)
        if self.mm_projector is not None:
            if self.is_internvl:
                vit_embeds = encoder_outputs[0]
                h = w = int(vit_embeds.shape[1] ** 0.5)
                vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
                vit_embeds = pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
                vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
                vit_embeds = vit_embeds.contiguous()
                output = self.mm_projector(vit_embeds)
            else:
                last_hidden_state = encoder_outputs[0]
                output = self.mm_projector(last_hidden_state)
        else:
            output = encoder_outputs[0]
        return output

class DflopPipeStage(TransformerDecoder):
    def __init__(self, llm_model, stage_id, vision_stages, llm_stages, vision_module, device, config, llm_config, model_dtype):
        print(f"[Rank : {torch.distributed.get_rank()}] Initializing DflopPipeStage with stage_id: {stage_id}, num_stages: {vision_stages + llm_stages}")
        model = llm_model
        tok_embeddings = model.tok_embeddings
        norm = model.norm
        output = model.output
        if stage_id in vision_stages:
            super().__init__(
                tok_embeddings=None,
                layers=nn.ModuleList(),
                max_seq_len=llm_model.max_seq_len,
                num_heads=llm_model.num_heads,
                head_dim=llm_model.head_dim,
                norm=None,
                output=None,
            )
        else:
            super().__init__(
                tok_embeddings=None,
                layers=model.layers,
                max_seq_len=model.max_seq_len,
                num_heads=model.num_heads,
                head_dim=model.head_dim,
                norm=model.norm,
                output=model.output,
            )
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.vision_stages = vision_stages
        self.llm_stages = llm_stages
        self.vision_module = None
        self.embed_tokens = None
        self.stage_id = stage_id
        print(f"Vision stage : {vision_stages} LLM stages : {llm_stages}, Stage id : {stage_id}")
        self.image_grid_pinpoints = config.image_grid_pinpoints
        self.image_size = config.image_size
        self.num_patches_per_side = config.num_patches_per_side
        self.tokenizer_model_max_length = config.tokenizer_model_max_length
        if stage_id in vision_stages:
            self.vision_module = vision_module
            self.layers = None
            self.output = None
            self.norm = None
        else:
            if stage_id == llm_stages[0]:
                self.image_newline = nn.Parameter(torch.empty(llm_config["embed_dim"], dtype=model_dtype))
                self.embed_tokens = tok_embeddings
                self.layers = model.layers
                self.output = None
                self.norm = None
            if stage_id == llm_stages[-1]:
                self.layers = model.layers
                self.norm = norm
                self.output = output
                self.loss_fn = LinearCrossEntropyLoss()
                self.loss_fn.set_model_output(model)            
            if (stage_id != llm_stages[0]) and (stage_id != llm_stages[-1]):
                self.layers = model.layers
                self.output = None
                self.norm = None
            self.device = device

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.reshape(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        height, width = image_feature.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.reshape(num_frames, -1, num_dim)
        return image_feature
    
    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.reshape(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature
    
    def postprocess_image_features(self, encoded_image_features, split_sizes, video_idx_in_batch, image_sizes):
        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        image_features = []
        for idx, image_feat in enumerate(encoded_image_features):
            if idx in video_idx_in_batch:
                image_features.append(self.get_2dPool(image_feat))
            else:
                image_features.append(image_feat)
        new_image_features = []
        for image_idx, image_feature in enumerate(image_features):
            if image_idx in video_idx_in_batch:  # video operations
                image_feature = self.add_token_per_grid(image_feature)
                new_image_features.append(image_feature)
            elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.num_patches_per_side
                assert height * width == base_image_feature.shape[0]
                max_num_patches = 9
                if self.image_size is not None:
                    vision_tower_image_size = self.image_size
                else:
                    raise ValueError("vision_tower_image_size is not found in the vision tower.")
                try:
                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.image_grid_pinpoints, vision_tower_image_size)
                except Exception as e:
                    num_patch_width, num_patch_height = 2, 2
                image_feature = image_feature.reshape(num_patch_height, num_patch_width, height, width, -1)
                

                unit = image_feature.shape[2]
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                c, h, w = image_feature.shape
                times = math.sqrt(h * w / (max_num_patches * unit**2))
                if times > 1.1:
                    image_feature = image_feature[None]
                    image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                new_image_features.append(image_feature)
            else:  # single image operations
                image_feature = image_feature[0]
                image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)

                new_image_features.append(image_feature)
        image_features = new_image_features
        return image_features
    
    def prepare_qwen_inputs(self, image_features, split_sizes, video_idx_in_batch, image_sizes, input_ids, labels, attention_mask, position_ids, past_key_values=None):
        image_features = self.postprocess_image_features(image_features, split_sizes, video_idx_in_batch, image_sizes)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
    

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        new_input_embeds = [x[:self.tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:self.tokenizer_model_max_length] for x in new_labels]

        packed_new_input_embeds = []
        packed_new_labels = []
        packed_position_ids = []

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            packed_new_input_embeds.append(cur_new_embed)
            packed_new_labels += [IGNORE_INDEX]+ cur_new_labels.tolist()[1:]
            packed_position_ids += list(range(len(cur_new_embed)))

        new_input_embeds = torch.concat(packed_new_input_embeds, dim=0).unsqueeze(0)
        new_labels = torch.tensor(packed_new_labels, dtype=torch.long if new_labels[0].dtype == torch.int32 else new_labels[0].dtype).unsqueeze(0)
        new_position_ids = torch.tensor(packed_position_ids, dtype=torch.long).unsqueeze(0)
        return new_input_embeds, new_position_ids, new_labels
        
    def forward(
        self,
        input_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[_MaskType] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs
        
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        # shape: [b, s, d]
        if self.stage_id in self.vision_stages:
            input_embeds = self.vision_module(input_embeds)
            return (input_embeds, )
        if self.stage_id == self.llm_stages[0]:
            split_sizes = kwargs['split_sizes']
            video_idx_in_batch = kwargs['video_idx_in_batch']
            image_sizes = kwargs['image_sizes']
            input_ids = kwargs['input_ids'].to(self.device)
            labels = kwargs['labels'].to(self.device)
            text_attention_mask = kwargs['text_attention_mask'].to(self.device)
            position_ids = None
            input_embeds, input_pos, _ = self.prepare_qwen_inputs(input_embeds, split_sizes, video_idx_in_batch, image_sizes, input_ids, labels, text_attention_mask, position_ids)

        h = input_embeds
        hidden = []
        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)
            # shape: [b, s, d]
            h = layer(
                h,
                mask=attention_mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
            )
        # shape: [b, seq_len, out_dim]
        if self.output:
            if len(self.layers) in self.output_hidden_states:
                hidden.append(h)
            output = self.norm(h)
            # Output list if hidden states are requested, otherwise just the output
            # TODO: always output a list to have a consistent output type
            output = output if not hidden else [*hidden, output]
            return output
        else:
            output = h
            return (output, )
