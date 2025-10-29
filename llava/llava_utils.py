import ast
import re
import math
import os
import torch
import transformers
import tokenizers

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
from PIL import ImageFile
from copy import deepcopy
from packaging import version
from torch import nn
from torch.utils.data import DataLoader
from transformers.modeling_outputs import BaseModelOutputWithPooling
from torchtune_models import flashqwen2
from torchtune.modules import TransformerDecoder
from torchtune.modules.attention_utils import _MaskType
from torchtune.modules.loss import LinearCrossEntropyLoss
from llava.constants import IGNORE_INDEX
from llava import conversation as conversation_lib
from llava.model import *
from llava.utils import rank0_print
from llava.model.llava_arch import unpad_image
from llava.mm_utils import get_anyres_image_grid_shape
from llava.train.train import *
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionModel
from llava.model.multimodal_projector.builder import build_vision_projector

torch.multiprocessing.set_sharing_strategy("file_system")
ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})

def init_qwen_loader_model(model_dtype):
    model_args = ModelArguments(model_name_or_path="/giant-data/team/4724/models/llava-onevision-qwen2-7b-ov",
                                version="qwen_2",
                                mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model",
                                vision_tower="google/siglip-so400m-patch14-384",
                                mm_projector_type="mlp2x_gelu",
                                mm_vision_select_layer=-2,
                                mm_use_im_start_end=False,
                                mm_use_im_patch_token=False,
                                mm_patch_merge_type="spatial_unpad"
                                )
    data_args = DataArguments(data_path="/giant-data/user/1113870/BDAI/dmllm_codes/LLaVA-NeXT/scripts/train/onevision.yaml",
                            lazy_preprocess=True,
                            image_folder="/giant-data/team/4724/datasets/llava_ov_data/llava_data",
                            image_aspect_ratio="anyres_max_9",
                            image_grid_pinpoints="(1x1),...,(6x6)",
                            video_folder="/giant-data/team/4724/datasets/llava_ov_data/llava_video",
                            frames_upbound=32)
    training_args= TrainingArguments(model_max_length=32768,
                                gradient_checkpointing=False,
                                group_by_modality_length=True,
                                mm_vision_tower_lr=2e-6,
                                output_dir="/giant-data/user/1113870/BDAI/LLaVA-NeXT/output",
                                per_device_train_batch_size=8,
                                learning_rate=1e-5,
                                num_train_epochs=1,
                                warmup_ratio=0.03,
                                gradient_accumulation_steps=1,
                                lr_scheduler_type="cosine",
                                bf16=True)
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16)

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    if data_args.image_grid_pinpoints is not None:
        if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
            try:
                patch_size = data_args.image_processor.size[0]
            except Exception as e:
                patch_size = data_args.image_processor.size["shortest_edge"]

            assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
            # Use regex to extract the range from the input string
            matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
            range_start = tuple(map(int, matches[0]))
            range_end = tuple(map(int, matches[-1]))
            # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
            grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
            # Multiply all elements by patch_size
            data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
        elif isinstance(data_args.image_grid_pinpoints, str):
            data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.image_crop_resolution = data_args.image_crop_resolution
    model.config.image_split_resolution = data_args.image_split_resolution
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_newline_position = model_args.mm_newline_position
    model.config.add_faster_video = model_args.add_faster_video
    model.config.faster_token_stride = model_args.faster_token_stride
    model.config.add_time_instruction = data_args.add_time_instruction
    model.config.force_sample = data_args.force_sample
    model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride 

    ### Deciding train which part of the model
    if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
        if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
            model.requires_grad_(False)
        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if model_args.tune_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
        if training_args.freeze_mm_vision_resampler:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = False

        model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
        if model_args.unfreeze_mm_vision_tower:
            vision_tower.requires_grad_(True)
        else:
            vision_tower.requires_grad_(False)

    else:
        model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
        # Set the entire model to not require gradients by default
        model.requires_grad_(False)
        vision_tower.requires_grad_(False)
        model.get_model().mm_projector.requires_grad_(False)
        model.get_model().vision_resampler.requires_grad_(False)
        # Parse the mm_tunable_parts to decide which parts to unfreeze
        tunable_parts = model_args.mm_tunable_parts.split(",")
        if "mm_mlp_adapter" in tunable_parts:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        if "mm_vision_resampler" in tunable_parts:
            for p in model.get_model().vision_resampler.parameters():
                p.requires_grad = True
        if "mm_vision_tower" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" in name:
                    param.requires_grad_(True)
        if "mm_language_model" in tunable_parts:
            for name, param in model.named_parameters():
                if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                    param.requires_grad_(True)

    total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
    trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer, model, model_dtype)
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    
    return tokenizer, train_dataset, data_collator, model, optimizer

class SigLipPipeStage(nn.Module):
    def __init__(self, base_model, stage_id, num_stages, device, model_num_layers=None):
        super().__init__()
        vit_model = deepcopy(base_model.get_vision_tower().vision_tower.vision_model)
        if model_num_layers is not None:
            vit_model.encoder.layers = nn.ModuleList(vit_model.encoder.layers[:model_num_layers])
        
        self.encoder = vit_model.encoder
        self.embeddings = None
        self.post_layernorm = None
        self.head = None
        # Model Architecture
            # embeddings : Embedding module
            # encoder : Transformer block
            # post_layernorm : Layer normalization after the transformer block
            # head : Final linear layer for classification
            # mm_projector : Projector for multimodal output
        # Model Split
            # [embeddings, self.encoder.layers[:(num_stage_per_rank * (i+1)) - 1]]
            # [self.encoder.layers[(num_stage_per_rank * (i)) - 1 : (num_stage_per_rank * (i+1)) - 1]]
            # [self.encoder.layers[(num_stage_per_rank * (i+1)) - 1 :], self.post_layernorm, self.head, self.mm_projector]
        self.device = device
        self.dtype = next(vit_model.parameters()).dtype
        # assert (
        #     (len(vit_model.encoder.layers) + 2) % num_stages == 0
        # ), f"model layers {len(vit_model.encoder.layers) + 2} must be evenly divisible by num_stages {num_stages}"
        layers_per_rank = (len(vit_model.encoder.layers) + 2) // num_stages
        if stage_id == 0:
            self.encoder.layers = nn.ModuleList(self.encoder.layers[:layers_per_rank-1])
            self.embeddings = vit_model.embeddings
        elif stage_id == num_stages - 1:
            self.encoder.layers = nn.ModuleList(self.encoder.layers[stage_id * layers_per_rank-1:])
            self.post_layernorm = vit_model.post_layernorm
            self.head = vit_model.head
            self.mm_projector = base_model.get_model().mm_projector
        else:
            self.encoder.layers = nn.ModuleList(self.encoder.layers[stage_id * layers_per_rank-1:(stage_id + 1) * layers_per_rank-1])

    def forward(self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # print(f"[Rank : {self.device.index}]Input : {pixel_values}")
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        assert isinstance(pixel_values, torch.Tensor) , f"images should be a Tensor, got {type(pixel_values)}"
        if self.embeddings is not None:
            hidden_states = self.embeddings(pixel_values)
        else:
            hidden_states = pixel_values

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        if self.post_layernorm is not None:
            last_hidden_state = self.post_layernorm(last_hidden_state)
            pooled_output = self.head(last_hidden_state)

            if not return_dict:
                output = (last_hidden_state, pooled_output) + encoder_outputs[1:]
            else:
                output = BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooled_output,
                    hidden_states=encoder_outputs.hidden_states,
                    attentions=encoder_outputs.attentions,
                )
            output = output.hidden_states[-1].to(pixel_values.dtype)
            output = self.mm_projector(output)
        else:
            output = last_hidden_state
        # print(f"Rank {self.device.index}, Input shape {output.shape}, Output shape: {output.shape}")
        assert output.shape[-2] == 729
        return output

# class QwenPipeStage(TransformerDecoder):
#     def __init__(self, pp_mesh, device, model_num_layers=28):
#         model = flashqwen2(
#             vocab_size=152064,
#             num_layers=model_num_layers,
#             num_heads=28,
#             num_kv_heads=4,
#             embed_dim=3584,
#             intermediate_dim=18944,
#             max_seq_len=32768,
#             attn_dropout=0.0,
#             norm_eps=1e-06,
#             rope_base=1000000.0,
#         )
#         # tok_embeddings = model.tok_embeddings
#         model_layers = model.layers
#         max_seq_len = model.max_seq_len
#         num_heads = model.num_heads
#         head_dim = model.head_dim
#         norm = model.norm
#         output = model.output
#         super().__init__(tok_embeddings=None,
#                         layers=model_layers,
#                         max_seq_len=max_seq_len,
#                         num_heads=num_heads,
#                         head_dim=head_dim,
#                         norm=norm,
#                         output=output,
#                         )
#         # Model Architecture
#             # tok_embeddings : Embedding module
#             # layers : Transformer block
#             # norm : Layer normalization after the transformer block
#             # output : Final linear layer for classification
#             # loss_fn : Loss function
#         # Model Split
#             # [tok_embeddings, self.encoder.layers[:(num_stage_per_rank * (i+1)) - 1]]
#             # [self.encoder.layers[(num_stage_per_rank * (i)) - 1 : (num_stage_per_rank * (i+1)) - 1]]
#             # [self.encoder.layers[(num_stage_per_rank * (i+1)) - 1 :], norm, output, loss_fn]
#         stage_id = pp_mesh.get_local_rank()
#         num_stages = pp_mesh.size()
#         assert (
#             (len(model.layers) + 2) % num_stages == 0
#         ), f"model layers {len(model.layers) + 2} must be evenly divisible by num_stages {num_stages}"
#         layers_per_rank = (len(model.layers) + 2) // num_stages
#         print(f"[Rank : {device.index}] Stage Id : {stage_id}, Num Stages : {num_stages}, Layers per Rank : {layers_per_rank}")
#         if stage_id == 0:
#             num_layers = layers_per_rank - 1
#             self.layers = nn.ModuleList(model.layers[:num_layers])
#             self.norm = None
#             self.output = None
#         elif stage_id == num_stages - 1:
#             num_layers = layers_per_rank
#             self.layers = nn.ModuleList(model.layers[stage_id * num_layers:])
#             self.norm = norm
#             self.output = output
#             self.loss_fn = LinearCrossEntropyLoss()
#             self.loss_fn.set_model_output(model)
#         else:
#             num_layers = layers_per_rank - 1
#             self.layers = nn.ModuleList(model.layers[stage_id * num_layers:(stage_id + 1) * num_layers])
#             self.norm = None
#             self.output = None
#         self.device = device
        
#     def forward(
#         self,
#         input_embeds: Optional[torch.Tensor] = None,
#         attention_mask: Optional[_MaskType] = None,
#         encoder_input: Optional[torch.Tensor] = None,
#         encoder_mask: Optional[torch.Tensor] = None,
#         input_pos: Optional[torch.Tensor] = None,
        
#     ) -> Union[torch.Tensor, list[torch.Tensor]]:
#         print(f"[Rank:{self.device.index}] Input : {input_embeds.shape}")
#         # shape: [b, s, d]
#         h = input_embeds
#         hidden = []
#         for i, layer in enumerate(self.layers):
#             if i in self.output_hidden_states:
#                 hidden.append(h)
#             # shape: [b, s, d]
#             h = layer(
#                 h,
#                 mask=attention_mask,
#                 encoder_input=encoder_input,
#                 encoder_mask=encoder_mask,
#                 input_pos=input_pos,
#             )
#         print(f"[Rank:{self.device.index}] h shape : {h.shape}")
#         # shape: [b, seq_len, out_dim]
#         if self.output:
#             if len(self.layers) in self.output_hidden_states:
#                 hidden.append(h)
#             output = self.norm(h)
#             # Output list if hidden states are requested, otherwise just the output
#             # TODO: always output a list to have a consistent output type
#             output = output if not hidden else [*hidden, output]
#             print(f"Rank {self.device} Input shape: {input_embeds.shape}, Output shape: {output.shape}")
#             return output
#         else:
#             output = h
#             # print(f"Rank {self.device} Input shape: {input_embeds.shape}, Output shape: {output.shape}")
#             return (output, )


class QwenPipeStage(TransformerDecoder):
    def __init__(self, base_model, stage_id, num_stages, device, model_num_layers=28):
        model = flashqwen2(
            vocab_size=152064,
            num_layers=model_num_layers,
            num_heads=28,
            num_kv_heads=4,
            embed_dim=3584,
            intermediate_dim=18944,
            max_seq_len=32768,
            attn_dropout=0.0,
            norm_eps=1e-06,
            rope_base=1000000.0,
        )
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
        # Model Architecture
            # tok_embeddings : Embedding module
            # layers : Transformer block
            # norm : Layer normalization after the transformer block
            # output : Final linear layer for classification
            # loss_fn : Loss function
        # Model Split
            # [tok_embeddings, self.encoder.layers[:(num_stage_per_rank * (i+1)) - 1]]
            # [self.encoder.layers[(num_stage_per_rank * (i)) - 1 : (num_stage_per_rank * (i+1)) - 1]]
            # [self.encoder.layers[(num_stage_per_rank * (i+1)) - 1 :], norm, output, loss_fn]
        # assert (
        #     (len(model.layers) + 2) % num_stages == 0
        # ), f"model layers {len(model.layers) + 2} must be evenly divisible by num_stages {num_stages}"
        layers_per_rank = (len(model.layers) + 2) // num_stages
        # layers_per_rank : 9 25 + 2
        # 0~7, 8~16, 17~24,
        # :8, 8:17, 17:25
        # :layer_idx-1 idx*(layer_idx)-1:(idx+1)*(layer_idx)-1 idx*(layer_idx)-1:
        # layers_per_rank : 8
        # 0~6, 7~14, 15~22, 23~29
        # :7, 7:15, 15:23, 23:30
        # :layer_idx-1 idx*(layer_idx)-1:(idx+1)*(layer_idx)-1

        print(f"[Rank : {device.index}] Stage Id : {stage_id}, Num Stages : {num_stages}, Layers per Rank : {layers_per_rank}")
        if stage_id == 0:
            self.layers = nn.ModuleList(model.layers[:layers_per_rank-1])
            self.norm = None
            self.output = None
            self.config = base_model.config
            self.get_2dPool = base_model.get_2dPool
            self.add_token_per_grid = base_model.add_token_per_grid
            self.num_patches_per_side = base_model.get_vision_tower().num_patches_per_side
            self.image_newline = base_model.model.image_newline
            self.image_size = base_model.get_vision_tower().image_size if hasattr(base_model.get_vision_tower(), "image_size") else None
            self.embed_tokens = tok_embeddings
        elif stage_id == num_stages - 1:
            self.layers = nn.ModuleList(model.layers[stage_id * layers_per_rank-1:])
            self.embed_tokens = None
            self.norm = norm
            self.output = output
            self.loss_fn = LinearCrossEntropyLoss()
            self.loss_fn.set_model_output(model)            
        else:
            self.layers = nn.ModuleList(model.layers[stage_id * layers_per_rank-1:(stage_id + 1) * layers_per_rank-1])
            self.embed_tokens = None
            self.norm = None
            self.output = None
        self.device = device

    def postprocess_image_features(self, encoded_image_features, split_sizes, video_idx_in_batch, image_sizes):
        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        image_features = []
        for idx, image_feat in enumerate(encoded_image_features):
            if idx in video_idx_in_batch:
                image_features.append(self.get_2dPool(image_feat))
            else:
                image_features.append(image_feat)
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

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

                if "anyres_max" in image_aspect_ratio:
                    matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                    if matched_anyres_max_num_patches:
                        max_num_patches = int(matched_anyres_max_num_patches.group(1))

                if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                    if self.image_size is not None:
                        vision_tower_image_size = self.image_size
                    else:
                        raise ValueError("vision_tower_image_size is not found in the vision tower.")
                    try:
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                    except Exception as e:
                        rank0_print(f"Error: {e}")
                        num_patch_width, num_patch_height = 2, 2
                    image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                else:
                    image_feature = image_feature.view(2, 2, height, width, -1)

                if "maxpool2x2" in mm_patch_merge_type:
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                    image_feature = nn.functional.max_pool2d(image_feature, 2)
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
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
                elif "unpad" in mm_patch_merge_type:
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                    image_feature = unpad_image(image_feature, image_sizes[image_idx])
                    image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                else:
                    image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                    image_feature = image_feature.flatten(0, 3)
                if "nobase" in mm_patch_merge_type:
                    pass
                else:
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                new_image_features.append(image_feature)
            else:  # single image operations
                image_feature = image_feature[0]
                if "unpad" in mm_patch_merge_type:
                    image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)

                new_image_features.append(image_feature)
        image_features = new_image_features
        return image_features
    
    def prepare_qwen_inputs(self, image_features, split_sizes, video_idx_in_batch, image_sizes, input_ids, labels, attention_mask, position_ids, past_key_values=None):
        image_features = self.postprocess_image_features(image_features, split_sizes, video_idx_in_batch, image_sizes)
        # input_ids = input_ids.to(self.device)
        # labels = labels.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
    
        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
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
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

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
        # print(f"[Rank:{self.device.index}] Input : {input_embeds.shape}")
        # shape: [b, s, d]
        if self.embed_tokens:
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
        # print(f"[Rank:{self.device.index}] h shape : {h.shape}")
        # shape: [b, seq_len, out_dim]
        if self.output:
            if len(self.layers) in self.output_hidden_states:
                hidden.append(h)
            output = self.norm(h)
            # Output list if hidden states are requested, otherwise just the output
            # TODO: always output a list to have a consistent output type
            output = output if not hidden else [*hidden, output]
            # print(f"Rank {self.device} Input shape: {input_embeds.shape}, Output shape: {output.shape}")
            return output
        else:
            output = h
            # print(f"Rank {self.device} Input shape: {input_embeds.shape}, Output shape: {output.shape}")
            return (output, )

class LLavaOVPipeStage(torch.nn.Module):
    def __init__(self, model, stage_id, num_stages, component, device, model_num_layers=None):
        super().__init__()
        self.stage_id = stage_id
        self.device = device
        self.component = component
        if component == "vision":
            self.model = SigLipPipeStage(model, stage_id, num_stages, device, model_num_layers)
        else:
            self.model = QwenPipeStage(model, stage_id, num_stages, device, model_num_layers)
    
    def forward(self, *args, **kwargs):
        if self.component == "vision":
            output = self.model(*args)
        else:
            output = self.model(*args, **kwargs)    
        return output


# class SigLipModule(nn.Module):
#     def __init__(self, base_model, device, model_num_layers=None):
#         super().__init__()
#         vit_model = deepcopy(base_model.get_vision_tower().vision_tower.vision_model)
#         if model_num_layers is not None:
#             vit_model.encoder.layers = nn.ModuleList(vit_model.encoder.layers[:model_num_layers])
        
#         self.encoder = vit_model.encoder
#         self.embeddings = vit_model.embeddings
#         # self.post_layernorm = vit_model.post_layernorm
#         # self.head = vit_model.head
#         self.mm_projector = base_model.get_model().mm_projector
#         self.device = device
#         self.dtype = next(vit_model.parameters()).dtype

class SigLipModule(nn.Module):
    def __init__(self, vision_config, mm_config, model_dtype, device):
        super().__init__()
        vit_model = SigLipVisionModel(vision_config).vision_model
        vit_model = vit_model.to(model_dtype)
        
        self.encoder = vit_model.encoder
        self.embeddings = vit_model.embeddings
        # self.post_layernorm = vit_model.post_layernorm
        # self.head = vit_model.head
        self.mm_projector = build_vision_projector(mm_config)
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

        # last_hidden_state = self.post_layernorm(last_hidden_state)
        # pooled_output = self.head(last_hidden_state)

        # if not return_dict:
        #     output = (last_hidden_state, pooled_output) + encoder_outputs[1:]
        # else:
        #     output = BaseModelOutputWithPooling(
        #         last_hidden_state=last_hidden_state,
        #         pooler_output=pooled_output,
        #         hidden_states=encoder_outputs.hidden_states,
        #         attentions=encoder_outputs.attentions,
        #     )
        # output = output.hidden_states[-1].to(pixel_values.dtype)
        # output = self.mm_projector(output)
        output = self.mm_projector(last_hidden_state)
        # if self.device.index == 0:
        #     print(f"[Rank : {torch.distributed.get_rank()}] SigLIP output shape {last_hidden_state.shape}, mm_projector output shape : {output.shape}")
        assert output.shape[-2] == 729
        return output

class BaseLLaVAOVPipeStage(TransformerDecoder):
    def __init__(self, base_model, qwen_model, stage_id, num_stages, device, vision_config, mm_config, llm_config, model_dtype):
        model = qwen_model
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
        self.stage_id = stage_id
        num_layers = llm_config["num_layers"]
        layers_per_rank = (num_layers + 2) // (num_stages - 1)
        transformer_layer = model.layers[0]
        remain_layers = (num_layers + 2) % (num_stages - 1)
        extra = 1 if 2 <= stage_id <= remain_layers else 0
        layers_per_rank = layers_per_rank + extra
        print(f"[Rank : {torch.distributed.get_rank()}] Stage Id : {stage_id}, Num Stages : {num_stages}, Layers per Rank : {layers_per_rank}")
        if stage_id == 0:
            self.vision_module = SigLipModule(vision_config, mm_config, model_dtype, device)
            self.layers = None
            self.config = base_model.config
            self.num_patches_per_side = base_model.get_vision_tower().num_patches_per_side
            self.image_newline = nn.Parameter(torch.empty(llm_config["embed_dim"], dtype=model_dtype))
            self.image_size = base_model.get_vision_tower().image_size if hasattr(base_model.get_vision_tower(), "image_size") else None
            self.embed_tokens = tok_embeddings
            self.output = None
            self.norm = None
        elif stage_id == num_stages - 1:
            layers = [deepcopy(transformer_layer) for _ in range(layers_per_rank)]
            self.layers = nn.ModuleList(layers)
            self.norm = norm
            self.output = output
            self.loss_fn = LinearCrossEntropyLoss()
            self.loss_fn.set_model_output(model)            
        else:
            layers = [deepcopy(transformer_layer) for _ in range(layers_per_rank)]
            self.layers = nn.ModuleList(layers)
            self.output = None
            self.norm = None
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
        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

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

                if "anyres_max" in image_aspect_ratio:
                    matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                    if matched_anyres_max_num_patches:
                        max_num_patches = int(matched_anyres_max_num_patches.group(1))

                if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                    if self.image_size is not None:
                        vision_tower_image_size = self.image_size
                    else:
                        raise ValueError("vision_tower_image_size is not found in the vision tower.")
                    try:
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                    except Exception as e:
                        rank0_print(f"Error: {e}")
                        num_patch_width, num_patch_height = 2, 2
                    image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                else:
                    image_feature = image_feature.view(2, 2, height, width, -1)

                if "maxpool2x2" in mm_patch_merge_type:
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                    image_feature = nn.functional.max_pool2d(image_feature, 2)
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
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
                elif "unpad" in mm_patch_merge_type:
                    image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                    image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                    image_feature = unpad_image(image_feature, image_sizes[image_idx])
                    image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                    image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                else:
                    image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                    image_feature = image_feature.flatten(0, 3)
                if "nobase" in mm_patch_merge_type:
                    pass
                else:
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                new_image_features.append(image_feature)
            else:  # single image operations
                image_feature = image_feature[0]
                if "unpad" in mm_patch_merge_type:
                    image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)

                new_image_features.append(image_feature)
        image_features = new_image_features
        return image_features
    
    def prepare_qwen_inputs(self, image_features, split_sizes, video_idx_in_batch, image_sizes, input_ids, labels, attention_mask, position_ids, past_key_values=None):
        image_features = self.postprocess_image_features(image_features, split_sizes, video_idx_in_batch, image_sizes)
        # input_ids = input_ids.to(self.device)
        # labels = labels.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
    
        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
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
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

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
        # print(f"[Rank:{self.device.index}] Input : {input_embeds.shape}")
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
            output, input_pos, _ = self.prepare_qwen_inputs(input_embeds, split_sizes, video_idx_in_batch, image_sizes, input_ids, labels, text_attention_mask, position_ids)
            return (output, )

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


# class LLavaOVPipeStage(torch.nn.Module):
#     def __init__(self, model, stage_id, num_stages, component, device, model_num_layers=None):
#         super().__init__()
#         self.stage_id = stage_id
#         self.qwen_stages = qwen_stages
#         self.device = device
#         if component == "vision":
#             self.model = SigLipPipeStage(model, stage_id, num_stages, device, model_num_layers)
#         else:
#             self.model = QwenPipeStage(stage_id, num_stages, device, model_num_layers)
#             if stage_id == 0:
#                 # For postprocessing
#                 self.config = model.config
#                 self.get_2dPool = model.get_2dPool
#                 self.add_token_per_grid = model.add_token_per_grid
#                 self.num_patches_per_side = model.get_vision_tower().num_patches_per_side
#                 self.image_newline = model.model.image_newline
#                 self.image_size = model.get_vision_tower().image_size if hasattr(model.get_vision_tower(), "image_size") else None
#                 self.embed_tokens = model.get_model().embed_tokens

#     def postprocess_image_features(self, encoded_image_features, split_sizes, video_idx_in_batch, image_sizes):
#         encoded_image_features = torch.split(encoded_image_features, split_sizes)
#         image_features = []
#         for idx, image_feat in enumerate(encoded_image_features):
#             if idx in video_idx_in_batch:
#                 image_features.append(self.get_2dPool(image_feat))
#             else:
#                 image_features.append(image_feat)
#         mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
#         image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

#         new_image_features = []
#         for image_idx, image_feature in enumerate(image_features):
#             if image_idx in video_idx_in_batch:  # video operations
#                 image_feature = self.add_token_per_grid(image_feature)
#                 new_image_features.append(image_feature)
#             elif image_feature.shape[0] > 1:  # multi patches and multi images operations
#                 base_image_feature = image_feature[0]
#                 image_feature = image_feature[1:]
#                 height = width = self.num_patches_per_side
#                 assert height * width == base_image_feature.shape[0]

#                 if "anyres_max" in image_aspect_ratio:
#                     matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
#                     if matched_anyres_max_num_patches:
#                         max_num_patches = int(matched_anyres_max_num_patches.group(1))

#                 if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
#                     if self.image_size is not None:
#                         vision_tower_image_size = self.image_size
#                     else:
#                         raise ValueError("vision_tower_image_size is not found in the vision tower.")
#                     try:
#                         num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
#                     except Exception as e:
#                         rank0_print(f"Error: {e}")
#                         num_patch_width, num_patch_height = 2, 2
#                     image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
#                 else:
#                     image_feature = image_feature.view(2, 2, height, width, -1)

#                 if "maxpool2x2" in mm_patch_merge_type:
#                     image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                     image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                     image_feature = nn.functional.max_pool2d(image_feature, 2)
#                     image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                 elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
#                     unit = image_feature.shape[2]
#                     image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                     image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                     image_feature = unpad_image(image_feature, image_sizes[image_idx])
#                     c, h, w = image_feature.shape
#                     times = math.sqrt(h * w / (max_num_patches * unit**2))
#                     if times > 1.1:
#                         image_feature = image_feature[None]
#                         image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
#                     image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
#                     image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                 elif "unpad" in mm_patch_merge_type:
#                     image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                     image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                     image_feature = unpad_image(image_feature, image_sizes[image_idx])
#                     image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
#                     image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                 else:
#                     image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
#                     image_feature = image_feature.flatten(0, 3)
#                 if "nobase" in mm_patch_merge_type:
#                     pass
#                 else:
#                     image_feature = torch.cat((base_image_feature, image_feature), dim=0)
#                 new_image_features.append(image_feature)
#             else:  # single image operations
#                 image_feature = image_feature[0]
#                 if "unpad" in mm_patch_merge_type:
#                     image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)

#                 new_image_features.append(image_feature)
#         image_features = new_image_features
#         return image_features
    
#     def prepare_qwen_inputs(self, image_features, split_sizes, video_idx_in_batch, image_sizes, input_ids, labels, attention_mask, position_ids, past_key_values=None):
#         image_features = self.postprocess_image_features(image_features, split_sizes, video_idx_in_batch, image_sizes)
#         # input_ids = input_ids.to(self.device)
#         # labels = labels.to(self.device)
#         # attention_mask = attention_mask.to(self.device)
#         # TODO: image start / end is not implemented here to support pretraining.
#         if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
#             raise NotImplementedError
#         _labels = labels
#         _position_ids = position_ids
#         _attention_mask = attention_mask
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
#         else:
#             attention_mask = attention_mask.bool()
#         if position_ids is None:
#             position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
#         if labels is None:
#             labels = torch.full_like(input_ids, IGNORE_INDEX)
    
#         # remove the padding using attention_mask -- FIXME
#         _input_ids = input_ids
#         input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
#         labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
#         new_input_embeds = []
#         new_labels = []
#         cur_image_idx = 0
#         # rank_print("Inserting Images embedding")
#         for batch_idx, cur_input_ids in enumerate(input_ids):
#             num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
#             # rank0_print(num_images)
#             if num_images == 0:
#                 cur_image_features = image_features[cur_image_idx]
#                 cur_input_embeds_1 = self.embed_tokens(cur_input_ids)
#                 cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
#                 new_input_embeds.append(cur_input_embeds)
#                 new_labels.append(labels[batch_idx])
#                 cur_image_idx += 1
#                 continue

#             image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
#             cur_input_ids_noim = []
#             cur_labels = labels[batch_idx]
#             cur_labels_noim = []
#             for i in range(len(image_token_indices) - 1):
#                 cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
#                 cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
#             split_sizes = [x.shape[0] for x in cur_labels_noim]
#             cur_input_embeds = self.embed_tokens(torch.cat(cur_input_ids_noim))
#             cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
#             cur_new_input_embeds = []
#             cur_new_labels = []

#             for i in range(num_images + 1):
#                 cur_new_input_embeds.append(cur_input_embeds_no_im[i])
#                 cur_new_labels.append(cur_labels_noim[i])
#                 if i < num_images:
#                     try:
#                         cur_image_features = image_features[cur_image_idx]
#                     except IndexError:
#                         cur_image_features = image_features[cur_image_idx - 1]
#                     cur_image_idx += 1
#                     cur_new_input_embeds.append(cur_image_features)
#                     cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

#             cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

#             # import pdb; pdb.set_trace()
#             cur_new_input_embeds = torch.cat(cur_new_input_embeds)
#             cur_new_labels = torch.cat(cur_new_labels)

#             new_input_embeds.append(cur_new_input_embeds)
#             new_labels.append(cur_new_labels)

#         # Truncate sequences to max length as image embeddings can make the sequence longer
#         tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
#         # rank_print("Finishing Inserting")

#         new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
#         new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

#         packed_new_input_embeds = []
#         packed_new_labels = []
#         packed_position_ids = []

#         for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
#             packed_new_input_embeds.append(cur_new_embed)
#             packed_new_labels += [IGNORE_INDEX]+ cur_new_labels.tolist()[1:]
#             packed_position_ids += list(range(len(cur_new_embed)))

#         new_input_embeds = torch.concat(packed_new_input_embeds, dim=0).unsqueeze(0)
#         new_labels = torch.tensor(packed_new_labels, dtype=torch.long if new_labels[0].dtype == torch.int32 else new_labels[0].dtype).unsqueeze(0)
#         new_position_ids = torch.tensor(packed_position_ids, dtype=torch.long).unsqueeze(0)
#         return new_input_embeds, new_position_ids, new_labels
    
#     def forward(self, *args, **kwargs):
#         if self.stage_id == self.qwen_stages[0]:
#             image_features = args[0]
#             split_sizes = kwargs['split_sizes']
#             video_idx_in_batch = kwargs['video_idx_in_batch']
#             image_sizes = kwargs['image_sizes']
#             input_ids = kwargs['input_ids'].to(self.device)
#             labels = kwargs['labels'].to(self.device)
#             attention_mask = kwargs['attention_mask'].to(self.device)
#             position_ids = kwargs['new_position_ids'].to(self.device)
#             inputs_embeds, new_position_ids, _ = self.prepare_qwen_inputs(image_features, split_sizes, video_idx_in_batch, image_sizes, input_ids, labels, attention_mask, position_ids)
#             # print(f"Input_ids shape {input_ids.shape}, Image features shape {image_features.shape}, Inputs_embeds shape {inputs_embeds.shape}, New Attention mask shape {new_attention_mask.shape}, Attention mask shape {attention_mask.shape}, New Labels shape {new_labels.shape}, Labels shape {labels.shape}")
#             output = self.model(inputs_embeds, input_pos=new_position_ids)
#         elif self.stage_id in self.qwen_stages:
#             input_embeds = args[0].to(self.device)
#             new_position_ids = kwargs.get('new_position_ids', None).to(self.device)
#             output = self.model(input_embeds, input_pos=new_position_ids)
#         else:
#             output = self.model(*args)
#         return output


# class QwenPipeStage(TransformerDecoder):
#     def __init__(self, stage_id, stages, device, model_num_layers=28):
#         model = flashqwen2(
#             vocab_size=152064,
#             num_layers=model_num_layers,
#             num_heads=28,
#             num_kv_heads=4,
#             embed_dim=3584,
#             intermediate_dim=18944,
#             max_seq_len=32768,
#             attn_dropout=0.0,
#             norm_eps=1e-06,
#             rope_base=1000000.0,
#         )
#         tok_embeddings = model.tok_embeddings
#         model_layers = model.layers
#         max_seq_len = model.max_seq_len
#         num_heads = model.num_heads
#         head_dim = model.head_dim
#         norm = model.norm
#         output = model.output
#         num_stages = len(stages)
#         num_layers = len(model.layers) // num_stages
#         super().__init__(tok_embeddings=tok_embeddings,
#                         layers=model_layers,
#                         max_seq_len=max_seq_len,
#                         num_heads=num_heads,
#                         head_dim=head_dim,
#                         norm=norm,
#                         output=output,
#                         )
#         local_id = stage_id - stages[0]
#         if stage_id == stages[0]:
#             self.layers = nn.ModuleList(model.layers[:num_layers])
#             self.tok_embeddings = tok_embeddings
#             self.norm = None
#             self.output = None
#         elif stage_id == stages[-1]:
#             self.layers = nn.ModuleList(model.layers[local_id * num_layers:])
#             self.tok_embeddings = None
#             self.norm = norm
#             self.output = output
#             self.loss_fn = LinearCrossEntropyLoss()
#             self.loss_fn.set_model_output(model)
#         else:
#             self.layers = nn.ModuleList(model.layers[local_id * num_layers:(local_id + 1) * num_layers])
#             self.tok_embeddings = None
#             self.norm = None
#             self.output = None
#         self.device = device
        
#     def forward(
#         self,
#         input_embeds: Optional[torch.Tensor] = None,
#         attention_mask: Optional[_MaskType] = None,
#         encoder_input: Optional[torch.Tensor] = None,
#         encoder_mask: Optional[torch.Tensor] = None,
#         input_pos: Optional[torch.Tensor] = None,
        
#     ) -> Union[torch.Tensor, list[torch.Tensor]]:
#         # print(f"[Rank:{self.device.index}] Input : {input_embeds}")
#         # shape: [b, s, d]
#         h = input_embeds
#         hidden = []
#         for i, layer in enumerate(self.layers):
#             if i in self.output_hidden_states:
#                 hidden.append(h)
#             # shape: [b, s, d]
#             h = layer(
#                 h,
#                 mask=attention_mask,
#                 encoder_input=encoder_input,
#                 encoder_mask=encoder_mask,
#                 input_pos=input_pos,
#             )

#         # shape: [b, seq_len, out_dim]
#         if self.output:
#             if len(self.layers) in self.output_hidden_states:
#                 hidden.append(h)
#             output = self.norm(h)
#             # Output list if hidden states are requested, otherwise just the output
#             # TODO: always output a list to have a consistent output type
#             output = output if not hidden else [*hidden, output]
#             # print(f"Rank {self.device} Input shape: {input_embeds.shape}, Output shape: {output.shape}")
#             return output
#         else:
#             output = h
#             # print(f"Rank {self.device} Input shape: {input_embeds.shape}, Output shape: {output.shape}")
#             return (output, )

# class LLavaOVPipeStage(torch.nn.Module):
#     def __init__(self, model, stage_id, vision_stages, qwen_stages, device, model_num_layers=None):
#         super().__init__()
#         self.stage_id = stage_id
#         self.qwen_stages = qwen_stages
#         self.device = device
#         if stage_id in vision_stages:
#             self.model = SigLipPipeStage(model, stage_id, vision_stages, device, model_num_layers)
#         else:
#             self.model = QwenPipeStage(stage_id, qwen_stages, device, model_num_layers)
#         if stage_id == qwen_stages[0]:
#             # For postprocessing
#             self.config = model.config
#             self.get_2dPool = model.get_2dPool
#             self.add_token_per_grid = model.add_token_per_grid
#             self.num_patches_per_side = model.get_vision_tower().num_patches_per_side
#             self.image_newline = model.model.image_newline
#             self.image_size = model.get_vision_tower().image_size if hasattr(model.get_vision_tower(), "image_size") else None
#             self.embed_tokens = model.get_model().embed_tokens

#     def postprocess_image_features(self, encoded_image_features, split_sizes, video_idx_in_batch, image_sizes):
#         encoded_image_features = torch.split(encoded_image_features, split_sizes)
#         image_features = []
#         for idx, image_feat in enumerate(encoded_image_features):
#             if idx in video_idx_in_batch:
#                 image_features.append(self.get_2dPool(image_feat))
#             else:
#                 image_features.append(image_feat)
#         mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
#         image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")

#         new_image_features = []
#         for image_idx, image_feature in enumerate(image_features):
#             if image_idx in video_idx_in_batch:  # video operations
#                 image_feature = self.add_token_per_grid(image_feature)
#                 new_image_features.append(image_feature)
#             elif image_feature.shape[0] > 1:  # multi patches and multi images operations
#                 base_image_feature = image_feature[0]
#                 image_feature = image_feature[1:]
#                 height = width = self.num_patches_per_side
#                 assert height * width == base_image_feature.shape[0]

#                 if "anyres_max" in image_aspect_ratio:
#                     matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
#                     if matched_anyres_max_num_patches:
#                         max_num_patches = int(matched_anyres_max_num_patches.group(1))

#                 if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
#                     if self.image_size is not None:
#                         vision_tower_image_size = self.image_size
#                     else:
#                         raise ValueError("vision_tower_image_size is not found in the vision tower.")
#                     try:
#                         num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
#                     except Exception as e:
#                         rank0_print(f"Error: {e}")
#                         num_patch_width, num_patch_height = 2, 2
#                     image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
#                 else:
#                     image_feature = image_feature.view(2, 2, height, width, -1)

#                 if "maxpool2x2" in mm_patch_merge_type:
#                     image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                     image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                     image_feature = nn.functional.max_pool2d(image_feature, 2)
#                     image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                 elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
#                     unit = image_feature.shape[2]
#                     image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                     image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                     image_feature = unpad_image(image_feature, image_sizes[image_idx])
#                     c, h, w = image_feature.shape
#                     times = math.sqrt(h * w / (max_num_patches * unit**2))
#                     if times > 1.1:
#                         image_feature = image_feature[None]
#                         image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
#                     image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
#                     image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                 elif "unpad" in mm_patch_merge_type:
#                     image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
#                     image_feature = image_feature.flatten(1, 2).flatten(2, 3)
#                     image_feature = unpad_image(image_feature, image_sizes[image_idx])
#                     image_feature = torch.cat((image_feature, self.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
#                     image_feature = image_feature.flatten(1, 2).transpose(0, 1)
#                 else:
#                     image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
#                     image_feature = image_feature.flatten(0, 3)
#                 if "nobase" in mm_patch_merge_type:
#                     pass
#                 else:
#                     image_feature = torch.cat((base_image_feature, image_feature), dim=0)
#                 new_image_features.append(image_feature)
#             else:  # single image operations
#                 image_feature = image_feature[0]
#                 if "unpad" in mm_patch_merge_type:
#                     image_feature = torch.cat((image_feature, self.image_newline[None]), dim=0)

#                 new_image_features.append(image_feature)
#         image_features = new_image_features
#         return image_features
    
#     def prepare_qwen_inputs(self, image_features, split_sizes, video_idx_in_batch, image_sizes, input_ids, labels, position_ids, attention_mask=None, past_key_values=None):
#         image_features = self.postprocess_image_features(image_features, split_sizes, video_idx_in_batch, image_sizes)
#         # input_ids = input_ids.to(self.device)
#         # labels = labels.to(self.device)
#         # attention_mask = attention_mask.to(self.device)
#         # TODO: image start / end is not implemented here to support pretraining.
#         if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
#             raise NotImplementedError
#         _labels = labels
#         _position_ids = position_ids
#         _attention_mask = attention_mask
#         if attention_mask is None:
#             attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
#         else:
#             attention_mask = attention_mask.bool()
#         if position_ids is None:
#             position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
#         if labels is None:
#             labels = torch.full_like(input_ids, IGNORE_INDEX)
    
#         # remove the padding using attention_mask -- FIXME
#         _input_ids = input_ids
#         input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
#         labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
#         new_input_embeds = []
#         new_labels = []
#         cur_image_idx = 0
#         # rank_print("Inserting Images embedding")
#         for batch_idx, cur_input_ids in enumerate(input_ids):
#             num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
#             # rank0_print(num_images)
#             if num_images == 0:
#                 cur_image_features = image_features[cur_image_idx]
#                 cur_input_embeds_1 = self.embed_tokens(cur_input_ids)
#                 cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
#                 new_input_embeds.append(cur_input_embeds)
#                 new_labels.append(labels[batch_idx])
#                 cur_image_idx += 1
#                 continue

#             image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
#             cur_input_ids_noim = []
#             cur_labels = labels[batch_idx]
#             cur_labels_noim = []
#             for i in range(len(image_token_indices) - 1):
#                 cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
#                 cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
#             split_sizes = [x.shape[0] for x in cur_labels_noim]
#             cur_input_embeds = self.embed_tokens(torch.cat(cur_input_ids_noim))
#             cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
#             cur_new_input_embeds = []
#             cur_new_labels = []

#             for i in range(num_images + 1):
#                 cur_new_input_embeds.append(cur_input_embeds_no_im[i])
#                 cur_new_labels.append(cur_labels_noim[i])
#                 if i < num_images:
#                     try:
#                         cur_image_features = image_features[cur_image_idx]
#                     except IndexError:
#                         cur_image_features = image_features[cur_image_idx - 1]
#                     cur_image_idx += 1
#                     cur_new_input_embeds.append(cur_image_features)
#                     cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

#             cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

#             # import pdb; pdb.set_trace()
#             cur_new_input_embeds = torch.cat(cur_new_input_embeds)
#             cur_new_labels = torch.cat(cur_new_labels)

#             new_input_embeds.append(cur_new_input_embeds)
#             new_labels.append(cur_new_labels)

#         # Truncate sequences to max length as image embeddings can make the sequence longer
#         tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
#         # rank_print("Finishing Inserting")

#         new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
#         new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

#         packed_new_input_embeds = []
#         packed_new_labels = []
#         packed_position_ids = []

#         for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
#             packed_new_input_embeds.append(cur_new_embed)
#             packed_new_labels += [IGNORE_INDEX]+ cur_new_labels.tolist()[1:]
#             packed_position_ids += list(range(len(cur_new_embed)))

#         new_input_embeds = torch.concat(packed_new_input_embeds, dim=0).unsqueeze(0)
#         new_labels = torch.tensor(packed_new_labels, dtype=torch.long if new_labels[0].dtype == torch.int32 else new_labels[0].dtype).unsqueeze(0)
#         new_position_ids = torch.tensor(packed_position_ids, dtype=torch.long).unsqueeze(0)
#         return new_input_embeds, new_position_ids, new_labels
    
#     def forward(self, *args, **kwargs):
#         if self.stage_id == self.qwen_stages[0]:
#             image_features = args[0]
#             split_sizes = kwargs['split_sizes']
#             video_idx_in_batch = kwargs['video_idx_in_batch']
#             image_sizes = kwargs['image_sizes']
#             input_ids = kwargs['input_ids'].to(self.device)
#             labels = kwargs['labels'].to(self.device)
#             position_ids = kwargs['new_position_ids'].to(self.device)
#             inputs_embeds, new_position_ids, _ = self.prepare_qwen_inputs(image_features, split_sizes, video_idx_in_batch, image_sizes, input_ids, labels, position_ids)
#             # print(f"Input_ids shape {input_ids.shape}, Image features shape {image_features.shape}, Inputs_embeds shape {inputs_embeds.shape}, New Attention mask shape {new_attention_mask.shape}, Attention mask shape {attention_mask.shape}, New Labels shape {new_labels.shape}, Labels shape {labels.shape}")
#             output = self.model(inputs_embeds, input_pos=new_position_ids)
#         elif self.stage_id in self.qwen_stages:
#             input_embeds = args[0].to(self.device)
#             new_position_ids = kwargs.get('new_position_ids', None).to(self.device)
#             output = self.model(input_embeds, input_pos=new_position_ids)
#         else:
#             output = self.model(*args)
#         return output