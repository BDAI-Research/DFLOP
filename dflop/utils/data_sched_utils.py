
import threading
import queue
import json
import time
import torch
from typing import List, Iterator
from torch.utils.data import Sampler
from llava.constants import IGNORE_INDEX
from dmllm_utils.data_utils import DataCollatorForSupervisedDataset

class IndexProducer(threading.Thread):
    def __init__(self, idx_path: str, q: queue.Queue, num_training_steps: int, module: str, dp_rank: int):
        super().__init__(daemon=True)
        self.idx_path = idx_path
        self.queue = q
        self.num_training_steps = num_training_steps
        self.module = module
        self.dp_rank = dp_rank

    def run(self):
        for iter_id in range(self.num_training_steps):
            batch_list = self._wait_and_load_data(iter_id)
            self.queue.put(batch_list)
        self.queue.put(None)

    def _wait_and_load_data(self, iter_id: int) -> List[List[int]]:
        while True:
            try:
                with open(self.idx_path, "r") as f:
                    idx_dict = json.load(f)
                if str(iter_id) in idx_dict:
                    if self.module == "vision":
                        return idx_dict[str(iter_id)]["vision"]
                    elif self.module == "llm":
                        return idx_dict[str(iter_id)]["llm"][str(self.dp_rank)]
                # print(f"[Producer] Waiting for iteration {iter_id} data...")
                time.sleep(1)
            except (FileNotFoundError, json.JSONDecodeError):
                # print(f"[Producer] Waiting for {self.idx_path} to be created...")
                time.sleep(2)

class QueueBatchSampler(Sampler[List[int]]):
    def __init__(self, q: queue.Queue):
        self.queue = q

    def __iter__(self) -> Iterator[List[int]]:
        while True:
            iteration_batches = self.queue.get()

            if iteration_batches is None:
                break
            for micro_batch in iteration_batches:
                yield micro_batch

    def __len__(self):
        raise NotImplementedError

class DataCollator(DataCollatorForSupervisedDataset):
    def __init__(self, config, llm_config, tokenizer, dtype, module, v_dp_size = None, dp_rank = None):
        super().__init__(config, llm_config, tokenizer, dtype)
        self.is_vision = False
        self.is_llm = False
        if module == "vision":
            self.is_vision = True
            self.v_dp_size = v_dp_size
            self.dp_rank = dp_rank
        elif module == "llm":
            self.is_llm = True
        else:
            raise ValueError(f"Unsupported module: {module}")
    def __call__(self, instances):
        if self.is_vision:
            split_sizes = []
            for instance in instances:
                images = [img[0] for img in instance['image']]
                if type(images) is list or images.ndim == 5:
                    if type(images) is list:
                        images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                    images_list = []
                    for image in images:
                        if image.ndim == 4:
                            images_list.append(image)
                        else:
                            images_list.append(image.unsqueeze(0))
                split_sizes.append(sum(len(img) for img in images_list))
            images = [instance["image"] for instance in instances]
            image_sizes = [im[1] for im_list in images for im in im_list]
            modalities = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            if isinstance(modalities, str):
                modalities = [modalities]
            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
                images_list = []
                for image in images:
                    if image.ndim == 4:
                        images_list.append(image)
                    else:
                        images_list.append(image.unsqueeze(0))

                concat_images = torch.cat([image for image in images_list], dim=0)
            total = concat_images.shape[0]
            base = total // self.v_dp_size
            remainder = total % self.v_dp_size
            split_img_size = [base + (1 if i < remainder else 0) for i in range(self.v_dp_size)]
            start = 0
            split_images = []
            for size in split_img_size:
                split_images.append(concat_images[start:start+size])
                start += size
            concat_images = split_images[self.dp_rank]
            batch = {"images" : concat_images, "split_sizes" : split_sizes}
        else:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
            labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = 0
            if len(input_ids) == 0:
                print(f"Input ids is None in index : {instances}")
            input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            labels=labels.long() if labels.dtype == torch.int32 else labels
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=attention_mask)
            images = [instance["image"] for instance in instances]
            image_sizes = [im[1] for im_list in images for im in im_list]
            modalities = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            if isinstance(modalities, str):
                modalities = [modalities]
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

                video_idx_in_batch = []
                for _ in range(len(modalities)):
                    if modalities[_] == "video":
                        video_idx_in_batch.append(_)

                images_list = []
                for image in images:
                    if image.ndim == 4:
                        images_list.append(image)
                    else:
                        images_list.append(image.unsqueeze(0))

                concat_images = torch.cat([image for image in images_list], dim=0)
                split_sizes = [image.shape[0] for image in images_list]
            batch["image_sizes"] = image_sizes
            batch["split_sizes"] = split_sizes
            batch["video_idx_in_batch"] = video_idx_in_batch
            fake_image_features = torch.zeros(concat_images.shape[0], self.image_feature_dim, self.embed_dim, dtype=self.dtype)
            fake_image_features = self.process_image_features(fake_image_features, split_sizes, video_idx_in_batch, image_sizes)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            _, new_position_ids, new_labels = self.process_llm_inputs(input_ids, attention_mask, labels, fake_image_features)
            batch["new_position_ids"] = new_position_ids
            batch["new_labels"] = new_labels
        return batch