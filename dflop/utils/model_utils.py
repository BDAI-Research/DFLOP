import os
import re
import ast
import math
import time
import json
import copy
import random
import yaml

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from packaging import version
from PIL import Image, ImageFile

import transformers
import tokenizers
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch.nn.functional as F
# LLaVA 관련 모듈
from llava.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX
)
from llava import conversation as conversation_lib
from llava.utils import rank0_print, process_video_with_decord
from llava.model.llava_arch import unpad_image
from llava.mm_utils import get_anyres_image_grid_shape, process_anyres_image
from llava.model.multimodal_encoder.siglip_encoder import (
    SigLipVisionModel,
    SigLipImageProcessor
)
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.train.train import preprocess
# torchtune 관련 모듈
from torchtune_models import flashqwen2
from torchtune.modules import TransformerDecoder
from torchtune.modules.attention_utils import _MaskType
from torchtune.modules.loss import LinearCrossEntropyLoss
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionConfig
from dmllm_utils.internvit_modules import InternVisionModel

def pixel_shuffle(x, scale_factor=0.5):
    x = x.contiguous()
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
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
        # self.post_layernorm = vit_model.post_layernorm
        # self.head = vit_model.head
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
        # print(f"[Rank : {self.device.index}]Input : {pixel_values}")
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
        # print(f"[Rank : {torch.distributed.get_rank()}] SigLIP output shape {last_hidden_state.shape}, mm_projector output shape : {output.shape}")
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
        # print(f"[Rank : {self.device.index}]Input : {pixel_values}")
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
        # print(f"[Rank : {self.device.index}]Input : {pixel_values}")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        print(f"[Rank : {torch.distributed.get_rank()}] vision input shape : {pixel_values.shape}")
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
        # print(f"[Rank : {torch.distributed.get_rank()}] SigLIP output shape {last_hidden_state.shape}, mm_projector output shape : {output.shape}")
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

        # print(f"ViT : {self.encoder}, mm_projector : {self.mm_projector}")
        
    def forward(self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # print(f"[Rank : {self.device.index}]Input : {pixel_values}")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        # print(f"[Rank : {torch.distributed.get_rank()}] SigLIP input shape : {pixel_values.shape}")
        assert isinstance(pixel_values, torch.Tensor) , f"images should be a Tensor, got {type(pixel_values)}"
        
        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(hidden_states)
        vit_embeds = encoder_outputs[0]
        # print(f"[ViT output] : {vit_embeds.shape}")
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = vit_embeds.contiguous()
        # print(f"[mm_projector input] : {vit_embeds.shape}")
        output = self.mm_projector(vit_embeds)
        # print(f"[mm_projector output] : {output.shape}")
        return output
    
class BasePipeStage(TransformerDecoder):
    def __init__(self, llm_model, stage_id, num_stages, vision_module, device, config, llm_config, model_dtype, encoder_split=True):
        model = llm_model
        tok_embeddings = model.tok_embeddings
        model_layers = model.layers
        max_seq_len = model.max_seq_len
        num_heads = model.num_heads
        head_dim = model.head_dim
        norm = model.norm
        output = model.output
        super().__init__(tok_embeddings=None,
                        layers=model_layers,
                        max_seq_len=max_seq_len,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        norm=norm,
                        output=output,
                        )
        self.vision_module = None
        self.embed_tokens = None
        # Configurations
        self.image_grid_pinpoints = config.image_grid_pinpoints
        self.image_size = config.image_size
        self.num_patches_per_side = config.num_patches_per_side
        self.tokenizer_model_max_length = config.tokenizer_model_max_length
        # print(f"[Rank : {torch.distributed.get_rank()}] Stage Id : {stage_id}, Num Stages : {num_stages}, Layers per Rank : {layers_per_rank}")        
        self.output = None
        self.norm = None
        self.stage_id = stage_id
        if stage_id == 0:
            self.vision_module = vision_module
            self.image_newline = nn.Parameter(torch.empty(llm_config["embed_dim"], dtype=model_dtype))
            self.embed_tokens = tok_embeddings
            if encoder_split:
                self.layers = None
            else:
                self.layers = model_layers
        else:
            self.layers = model_layers
            if stage_id == num_stages - 1:
                self.norm = norm
                self.output = output
                self.loss_fn = LinearCrossEntropyLoss()
                self.loss_fn.set_model_output(model)            
        self.device = device
        # print(f"[Rank : {torch.distributed.get_rank()}] Model Initialization finished")

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        height, width = image_feature.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature
    
    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
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
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                

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
        # rank_print("Inserting Images embedding")
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

            # import pdb; pdb.set_trace()
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
        # print(f"[Rank : {torch.distributed.get_rank()}] Input shape: {input_embeds.shape}")
        # shape: [b, s, d]
        if self.stage_id == 0:
            input_embeds = self.vision_module(input_embeds)
            split_sizes = kwargs['split_sizes']
            video_idx_in_batch = kwargs['video_idx_in_batch']
            image_sizes = kwargs['image_sizes']
            input_ids = kwargs['input_ids'].to(self.device)
            labels = kwargs['labels'].to(self.device)
            text_attention_mask = kwargs['text_attention_mask'].to(self.device)
            position_ids = None
            input_embeds, input_pos, _ = self.prepare_qwen_inputs(input_embeds, split_sizes, video_idx_in_batch, image_sizes, input_ids, labels, text_attention_mask, position_ids)
            # print(f"[Rank : {torch.distributed.get_rank()}] Input shape: {input_embeds.shape}, Output shape: {output.shape}")
            if self.layers is None:
                return input_embeds
            else:
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
                output = h
                # print(f"[Rank : {torch.distributed.get_rank()}] Input shape: {input_embeds.shape}, Output shape: {output.shape}")
                return (output, )
        # return (output, )
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
        # print(f"[Rank:{self.device.index}] h shape : {h.shape}")
        # shape: [b, seq_len, out_dim]
        if self.output:
            if len(self.layers) in self.output_hidden_states:
                hidden.append(h)
            output = self.norm(h)
            # Output list if hidden states are requested, otherwise just the output
            # TODO: always output a list to have a consistent output type
            output = output if not hidden else [*hidden, output]
            # print(f"[Rank : {torch.distributed.get_rank()}] Input shape: {input_embeds.shape}, Output shape: {output.shape}")
            return output
        else:
            output = h
            # print(f"[Rank : {torch.distributed.get_rank()}] Input shape: {input_embeds.shape}, Output shape: {output.shape}")
            return (output, )