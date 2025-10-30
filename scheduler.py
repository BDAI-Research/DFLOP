import os
import json
import threading
import numpy as np
import transformers
import torch
import yaml
import numpy as np
from ortools.sat.python import cp_model
from typing import Dict, Sequence
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, RandomSampler
from llava.constants import IGNORE_INDEX
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from dflop.data import TrainConfig, LazySupervisedDataset, DataCollatorForSupervisedDataset, SigLipImageProcessor
from dflop.model import llm_configs, vision_configs
from dflop.prof_utils import vision_module_flops, llm_module_flops
from dflop.config import (
    get_config_path,
    load_config,
    resolve_path,
    reset_config_cache,
)


class IlpDataCollator(DataCollatorForSupervisedDataset):
    # Class-level lock for file operations
    _file_lock = threading.Lock()

    def __init__(self, config, image_size, vocab_size, parallel_config, vision_config, llm_config, tokenizer, dtype, num_batches, sched_data_path, scale_factor=10000):
        super().__init__(config, llm_config, tokenizer, dtype)
        self.num_batches = num_batches
        self.image_size = image_size
        
        self.vision_hidden_size = vision_config.hidden_size
        self.vision_intermediate_size = vision_config.intermediate_size
        self.llm_hidden_size = llm_config["embed_dim"]
        
        num_kv_heads = llm_config["num_kv_heads"]
        self.attention_heads = llm_config["num_heads"]
        self.query_groups = self.attention_heads // num_kv_heads
        self.ffn_hidden_size = llm_config["intermediate_dim"]
        self.vocab_size = vocab_size

        self.v_thr = parallel_config["vision_thr"]
        self.v_tp_size = parallel_config["vision_tp_size"]
        self.v_dp_size = parallel_config["vision_dp_size"]
        self.v_pp_size = parallel_config["vision_pp_size"]
        self.l_thr = parallel_config["llm_thr"]
        self.l_tp_size = parallel_config["llm_tp_size"]
        self.l_dp_size = parallel_config["llm_dp_size"]
        self.l_pp_size = parallel_config["llm_pp_size"]

        self.scale_factor = scale_factor
        self.sched_data_path = sched_data_path

        self.vision_num_layers = vision_config.num_hidden_layers
        self.llm_num_layers = llm_config["num_layers"]

    def get_data_dur_info(self, image_list, pos_ids_list):
        data_info = []
        for img, pos_ids in zip(image_list, pos_ids_list):
            img_batch_size = img.shape[0]
            llm_input_seq_len = pos_ids.shape[1]
            vision_flops = vision_module_flops(img_batch_size, self.image_size, self.vision_num_layers, self.vision_hidden_size, self.vision_intermediate_size, self.llm_hidden_size)/1e12
            llm_flops = llm_module_flops(llm_input_seq_len, 1, self.llm_hidden_size, self.llm_num_layers, self.query_groups, self.attention_heads, self.ffn_hidden_size, self.vocab_size, act_recomp=True)/1e12
            vision_dur = int(vision_flops / (self.v_thr * self.v_dp_size * self.v_pp_size * self.v_tp_size) * self.scale_factor)
            llm_dur = int(llm_flops / (self.l_thr * self.l_dp_size * self.l_pp_size * self.l_tp_size) * self.scale_factor)
            data_info.append([vision_dur, llm_dur])
        return data_info

    def ilp_data_sched(self, data_info, search_time_limit=10):
        N = len(data_info)
        m = self.l_dp_size * self.num_batches
        model = cp_model.CpModel()
        x = {}
        for i in range(N):
            for j in range(m):
                    x[i, j] = model.NewBoolVar(f'x_{i}_{j}')
        for i in range(N):
            model.Add(sum(x[i, j] for j in range(m)) == 1)

        for j in range(m):
            model.Add(sum(x[i, j] for i in range(N)) >= 1)

        upper_bound = sum(sum(data) for data in data_info)

        Global_C_max = model.NewIntVar(0, upper_bound, 'Global_C_max')
        for j in range(m):
            for c in range(len(data_info[0])):
                total_cost = sum(data_info[i][c] * x[i, j] for i in range(N))
                model.Add(total_cost <= Global_C_max)

        model.Minimize(Global_C_max)
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = 42
        solver.parameters.max_time_in_seconds = search_time_limit

        status = solver.Solve(model)
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            final_distribution = {j: [] for j in range(m)}
            for i in range(N):
                for j in range(m):
                    if solver.Value(x[i, j]) == 1:
                        final_distribution[j].append(i)
                        break
            data_idx_list = list(final_distribution.values())
        else:
            print("Trying Data scheduling again")
            model = cp_model.CpModel()
            x = {}
            for i in range(N):
                for j in range(m):
                        x[i, j] = model.NewBoolVar(f'x_{i}_{j}')
            for i in range(N):
                model.Add(sum(x[i, j] for j in range(m)) == 1)

            for j in range(m):
                model.Add(sum(x[i, j] for i in range(N)) >= 1)

            upper_bound = sum(sum(data) for data in data_info)

            Global_C_max = model.NewIntVar(0, upper_bound, 'Global_C_max')
            for j in range(m):
                for c in range(len(data_info[0])):
                    total_cost = sum(data_info[i][c] * x[i, j] for i in range(N))
                    model.Add(total_cost <= Global_C_max)

            model.Minimize(Global_C_max)
            solver = cp_model.CpSolver()
            solver.parameters.random_seed = 42
            solver.parameters.max_time_in_seconds = search_time_limit * 10

            status = solver.Solve(model)
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                final_distribution = {j: [] for j in range(m)}
                for i in range(N):
                    for j in range(m):
                        if solver.Value(x[i, j]) == 1:
                            final_distribution[j].append(i)
                            break
                data_idx_list = list(final_distribution.values())
            else:
                raise ValueError("No solution found for data scheduling.")
        return data_idx_list
    
    def get_scheduled_data(self, image_info_list, input_ids, attention_mask, labels):
        image_list = []
        image_size_list = []
        split_size_list = []
        video_idx_in_batch_list = []
        input_ids_list = []
        attention_mask_list = []
        new_labels_list = []
        new_position_ids_list = []

        for img, inp_ids, mask, label in zip(image_info_list, input_ids, attention_mask, labels):
            inp_ids = inp_ids.unsqueeze(0)
            mask = mask.unsqueeze(0)
            label = label.unsqueeze(0)

            image_sizes = [im[1] for im in img]
            modalities = [im[2] for im in img]
            images = [im[0] for im in img]
            if isinstance(modalities, str):
                modalities = [modalities]
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

                images_list = []
                for image in images:
                    if image.ndim == 4:
                        images_list.append(image)
                    else:
                        images_list.append(image.unsqueeze(0))
                
                video_idx_in_batch = []
                for _ in range(len(modalities)):
                    if modalities[_] == "video":
                        video_idx_in_batch.append(_)
                concat_images = torch.cat([image for image in images_list], dim=0)
                split_sizes = [image.shape[0] for image in images_list]
                fake_image_features = torch.zeros(concat_images.shape[0], self.image_feature_dim, self.embed_dim, dtype=self.dtype)
                fake_image_features = self.process_image_features(fake_image_features, split_sizes, video_idx_in_batch, image_sizes)
                _, new_position_ids, new_labels = self.process_llm_inputs(inp_ids, mask, label, fake_image_features)

                image_list.append(concat_images)
                image_size_list.append(image_sizes)
                split_size_list.append(split_sizes)
                video_idx_in_batch_list.append(video_idx_in_batch)
                input_ids_list.append(inp_ids)
                attention_mask_list.append(mask)
                new_labels_list.append(new_labels)
                new_position_ids_list.append(new_position_ids)
        data_info = self.get_data_dur_info(image_list, new_position_ids_list)
        sched_data_idx = self.ilp_data_sched(data_info)
        return sched_data_idx

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        idx = [instance["id"] for instance in instances]
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        labels=labels.long() if labels.dtype == torch.int32 else labels
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        image_info_list = [instance["image"] for instance in instances]
        sched_data_idx = self.get_scheduled_data(image_info_list, input_ids, attention_mask, labels)
        sched_vision_idx_list = []
        sched_llm_idx_list = [[] for _ in range(self.l_dp_size)]
        idx_list = []
        for i, group_idx in enumerate(sched_data_idx):
            data_idx = [idx[i] for i in group_idx]
            idx_list += data_idx
            if (i+1) % self.l_dp_size == 0:
                sched_vision_idx_list.append(idx_list)
                idx_list = []
            sched_llm_idx_list[i % self.l_dp_size].append(data_idx)
        new_data_dict = {"vision" : sched_vision_idx_list, "llm" : {dp_rank : sched_llm_idx_list[dp_rank] for dp_rank in range(self.l_dp_size)}}

        with IlpDataCollator._file_lock:
            if not os.path.exists(self.sched_data_path):
                data_dict = {0 : new_data_dict}
                with open(self.sched_data_path, "w") as f:
                    json.dump(data_dict, f, indent=2)
            else:
                with open(self.sched_data_path, "r") as f:
                    data_dict = json.load(f)
                iter_id = len(data_dict.keys())
                data_dict[iter_id] = new_data_dict
                with open(self.sched_data_path, "w") as f:
                    json.dump(data_dict, f, indent=2)

if __name__ == "__main__":
    print("="*30, "Start Data Scheduling", "="*30)
    default_config_path = get_config_path()
    os.environ.setdefault("DFLOP_CONFIG", str(default_config_path))
    reset_config_cache()
    root_config = load_config()

    paths_cfg = root_config.get("paths", {})
    model_paths_cfg = paths_cfg.get("model_dir", {})
    result_path = str(resolve_path(paths_cfg.get("result_dir")))
    data_sched_config_path = f"{result_path}/data_sched_config.yaml"

    with open(data_sched_config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    num_batches = config["num_batches"]
    parallel_config = config["parallel_config"]

    vision_model_name = config["vision_model_name"]
    vision_model_size = config["vision_model_size"]
    llm_model_name = config["llm_model_name"]
    llm_model_size = config["llm_model_size"]
    data_file_path = f"{result_path}/sched_data_idx.json"
    num_training_steps = config["num_training_steps"]
    dataset_path = resolve_path(paths_cfg.get("dataset_yaml"))
    image_folder = resolve_path(paths_cfg.get("image_folder"))
    video_folder = resolve_path(paths_cfg.get("video_folder"))

    required_paths = {
        "paths.dataset_yaml": dataset_path,
        "paths.image_folder": image_folder,
        "paths.video_folder": video_folder,
    }
    missing_paths = [key for key, value in required_paths.items() if value is None]
    if missing_paths:
        raise ValueError(f"Missing scheduler configuration paths: {', '.join(missing_paths)}")

    dataset_path_str = str(dataset_path)
    image_folder_str = str(image_folder)
    video_folder_str = str(video_folder)

    if vision_model_name == "siglip":
        image_size = 384
        image_processor = SigLipImageProcessor()
    elif vision_model_name == "internvit":
        image_size = 448
        image_processor = SigLipImageProcessor(size=(image_size, image_size), crop_size={"height": image_size, "width": image_size})
    else:
        raise ValueError(f"Unsupported vision model: {vision_model_name}")

    model_path_resolved = resolve_path(model_paths_cfg.get(llm_model_name))
    if model_path_resolved is None:
        raise ValueError(f"Missing model path for '{llm_model_name}' in model_paths configuration.")
    model_path = str(model_path_resolved)

    if llm_model_name == "qwen2.5":
        vocab_size = 152064
    elif llm_model_name == "llama3":
        vocab_size = 128256
    else:
        raise ValueError(f"Unsupported LLM model: {llm_model_name}")
    
    vision_config = vision_configs[vision_model_name][vision_model_size]
    llm_config = llm_configs[llm_model_name][llm_model_size]
    vision_num_layers = vision_config.num_hidden_layers
    llm_num_layers = llm_config["num_layers"]

    # LLM configuration
    gbs = 1
    hidden_size = llm_config["embed_dim"]
    num_kv_heads = llm_config["num_kv_heads"]
    attention_heads = llm_config["num_heads"]
    query_groups = attention_heads // num_kv_heads
    ffn_hidden_size = llm_config["intermediate_dim"]

    patch_size = 14
    in_channels = 3
    img_h = image_size
    img_w = image_size
    patch_dim = patch_size
    hs = vision_config.hidden_size
    intermediate_size = vision_config.intermediate_size
    llm_config = llm_configs[llm_model_name][llm_model_size]
    llm_hidden_size = llm_config["embed_dim"]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side="right")
    tokenizer_model_max_length = tokenizer.model_max_length
    train_config = TrainConfig(image_folder=image_folder_str,
                    video_folder=video_folder_str,
                    image_processor=image_processor,
                    tokenizer_model_max_length=tokenizer_model_max_length,
                    image_size=image_size,
                    patch_size=patch_size,
                )
    model_dtype = torch.float16
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = LazySupervisedDataset(dataset_path_str, tokenizer, train_config)

    global_batch_size = config["global_batch_size"]
    data_collator = IlpDataCollator(train_config, image_size, vocab_size, parallel_config, vision_config, llm_config, tokenizer, model_dtype, num_batches, data_file_path)
    sampler = RandomSampler(dataset, generator=generator)
    dataloader = DataLoader(dataset, batch_size=global_batch_size, sampler=sampler, collate_fn=data_collator, num_workers=8)
    data_iter = iter(dataloader)
    for step in range(num_training_steps):
        next(data_iter)
