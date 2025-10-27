# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

import copy
import csv
import itertools
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule, UnshardHandle
from torch.profiler import record_function
from .dflop_pp_stage_module import _RecvInfo
from torch.distributed.pipelining.microbatch import merge_chunks, split_args_kwargs_into_chunks, TensorChunkSpec
from torch.distributed.pipelining.stage import _PipelineStageBase
from utils.profiler import  AllocatedMemContext

if TYPE_CHECKING:
    from torch.distributed import Work

__all__ = [
    "get_schedule_class",
    "PipelineScheduleSingle",
    "PipelineScheduleMulti",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
    "ScheduleInterleavedZeroBubble",
    "ScheduleZBVZeroBubble",
]

logger = logging.getLogger(__name__)


class _ComputationType(Enum):
    # TODO(whc) rename to _ActType?
    FORWARD = 1
    BACKWARD_INPUT = 2
    BACKWARD_WEIGHT = 3
    UNSHARD = 4
    RESHARD = 5
    SEND_F = 6
    RECV_F = 7
    SEND_B = 8
    RECV_B = 9
    FULL_BACKWARD = 10

    def __str__(self):
        str_map = {
            _ComputationType.FORWARD: "F",
            _ComputationType.BACKWARD_INPUT: "I",
            _ComputationType.BACKWARD_WEIGHT: "W",
            _ComputationType.UNSHARD: "UNSHARD",
            _ComputationType.RESHARD: "RESHARD",
            _ComputationType.SEND_F: "SEND_F",
            _ComputationType.RECV_F: "RECV_F",
            _ComputationType.SEND_B: "SEND_B",
            _ComputationType.RECV_B: "RECV_B",
            _ComputationType.FULL_BACKWARD: "B",
        }
        return str_map[self]

    @staticmethod
    def from_str(action):
        if action == "F":
            return _ComputationType.FORWARD
        elif action == "I":
            return _ComputationType.BACKWARD_INPUT
        elif action == "W":
            return _ComputationType.BACKWARD_WEIGHT
        elif action == "UNSHARD":
            return _ComputationType.UNSHARD
        elif action == "RESHARD":
            return _ComputationType.RESHARD
        elif action == "SEND_F":
            return _ComputationType.SEND_F
        elif action == "RECV_F":
            return _ComputationType.RECV_F
        elif action == "SEND_B":
            return _ComputationType.SEND_B
        elif action == "RECV_B":
            return _ComputationType.RECV_B
        elif action == "B":
            return _ComputationType.FULL_BACKWARD
        else:
            raise RuntimeError(f"Invalid computation type {action}")


FORWARD = _ComputationType.FORWARD
BACKWARD_INPUT = _ComputationType.BACKWARD_INPUT
BACKWARD_WEIGHT = _ComputationType.BACKWARD_WEIGHT
UNSHARD = _ComputationType.UNSHARD
RESHARD = _ComputationType.RESHARD
SEND_F = _ComputationType.SEND_F
RECV_F = _ComputationType.RECV_F
SEND_B = _ComputationType.SEND_B
RECV_B = _ComputationType.RECV_B
FULL_BACKWARD = _ComputationType.FULL_BACKWARD

# Convenience shorthand for compute actions only since they are used in 'simple schedule format'
F = FORWARD
I = BACKWARD_INPUT
W = BACKWARD_WEIGHT
B = FULL_BACKWARD

# Helper to parse an action string like 1F0 into a tuple of (stage_index, computation_type, microbatch_index)
_action_regex = re.compile(
    r"(\d+)(F|I|B|W|UNSHARD|RESHARD|SEND_F|RECV_F|SEND_B|RECV_B)(\d*)"
)

def _make_tensor_from_meta(
    example: Union[torch.Size, torch.dtype],
) -> torch.Tensor:
    """
    Create a real tensor from a tensor.
    """
    return torch.empty(
        example[0],
        dtype=example[1],
    )

class _RootArgPlaceholder:
    """
    Placeholder for model-level inputs.
    """

    def __init__(self, example):
        self.meta = torch.empty(
                    tuple(example[0]),
                    dtype=example[1],    
                    ).to("meta")

class _Action(NamedTuple):
    stage_index: int
    computation_type: _ComputationType
    microbatch_index: Optional[int] = None

    def __repr__(self):
        repr = str(self.stage_index)
        repr += str(self.computation_type)
        if self.microbatch_index is not None:
            repr += str(self.microbatch_index)
        return repr

    @staticmethod
    def from_str(action_string: str):
        """
        Reverse of __repr__

        String should be formatted as [stage][action type][(microbatch)]
            e.g. `2F0`, `1UNSHARD`, `3SEND_F1`
        """
        action_string = action_string.strip()
        if match := _action_regex.match(action_string):
            stage_index, computation_type, microbatch_index = match.groups()
            return _Action(
                int(stage_index),
                _ComputationType.from_str(computation_type),
                int(microbatch_index) if len(microbatch_index) else None,
            )
        elif action_string == "":
            return None
        raise RuntimeError(
            f"Invalid action string: {action_string}, should be formatted as [stage][action type][(microbatch)] e.g. 2F0"
        )


def _format_pipeline_order(
    pipeline_order: Dict[int, List[Optional[_Action]]],
    error_step_number: Optional[int] = None,
) -> str:
    """
    Formats the pipeline order in a timestep (row) x rank (column) grid of actions
    and returns the formatted string.

    If `error_step_number` is passed in, an additional label will be added to signify which step
    that it is erroring on.
    """

    # don't mutate the original
    pipeline_order = copy.deepcopy(pipeline_order)

    # Replace None with ""
    for rank in pipeline_order:
        for i in range(len(pipeline_order[rank])):
            if pipeline_order[rank][i] is None:
                # TODO make a real 'None action' that prints as empty string and make mypy happy
                pipeline_order[rank][i] = ""  # type: ignore[call-overload]

    # Calculate the maximum number of steps across all ranks
    num_steps = max(len(actions) for actions in pipeline_order.values())
    step_labels = [
        "Step " + str(i).zfill(len(str(num_steps - 1))) for i in range(num_steps)
    ]
    # Sorting the dictionary by keys and ㅔing values in that order
    rank_actions = [
        pipeline_order.get(key, [""] * num_steps) for key in sorted(pipeline_order)
    ]
    # Transpose the list of lists (rows to columns)
    transposed_actions = list(itertools.zip_longest(*rank_actions, fillvalue=""))
    # Generate column labels for ranks
    num_ranks = len(pipeline_order)
    rank_labels = ["Rank " + str(i) for i in range(num_ranks)]
    # Calculate the maximum length of each column, considering labels
    max_lengths = [
        max(len(str(item)) if item is not None else 0 for item in col)
        for col in zip(step_labels, *transposed_actions)
    ]
    # Format the header row with rank labels
    header_row = " " * (len(step_labels[0]) + 2) + " ".join(
        f"{label:<{max_lengths[i]}}" for i, label in enumerate(rank_labels)
    )
    # Format each row with its corresponding label
    formatted_rows = [
        f"{label}: "
        + " ".join(f"{str(item):<{max_lengths[i]}}" for i, item in enumerate(row))
        + (
            " <-- ERROR HERE"
            if error_step_number is not None
            and int(label.split()[1]) == error_step_number
            else ""
        )
        for label, row in zip(step_labels, transposed_actions)
    ]
    # Join the rows into a single string
    formatted_table = header_row + "\n" + "\n".join(formatted_rows) + "\n"
    return formatted_table


class _PipelineSchedule(ABC):
    def __init__(
        self,
        n_microbatches: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # From arguments
        self._n_microbatches = n_microbatches
        self._loss_fn = loss_fn
        # Chunking specification for positional inputs. (default: `None`)
        self._args_chunk_spec = args_chunk_spec
        # Chunking specification for keyword inputs. (default: `None`)
        self._kwargs_chunk_spec = kwargs_chunk_spec
        self._output_merge_spec = output_merge_spec
        """
        # args_chunk_spec and kwargs_chunk_spec specify how to chunk inputs.
        # They are used to convert batch to microbatches in `step(x)`.  See
        # `TensorChunkSpec` for helper methods for creating them.
        """

        # Derived
        self._has_backward = self._loss_fn is not None

        # Holds the losses for each microbatch.
        self._internal_losses: List[torch.Tensor] = []
        logger.info("Using %s", self.__class__.__name__)

    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        if stage.is_last and self._stage.is_llm and self._has_backward:
            # print(f"Rank {stage.device} target_mbs : {len(target_mbs)}, computing loss for microbatch {mb_index}")
            loss = self._compute_loss(output, target_mbs[mb_index])  # type: ignore[index]
            self._internal_losses.append(loss)

    def _maybe_get_loss(self, stage, mb_index):
        valid_index = 0 <= mb_index < len(self._internal_losses)
        if stage.is_last and self._has_backward and valid_index:
            return self._internal_losses[mb_index]
        elif len(self._internal_losses) != 0 and not valid_index:
            raise RuntimeError(
                f"Loss for microbatch {mb_index} is not available. "
                f"Available losses for microbatches: {self._internal_losses}"
            )
        else:
            return None

    def _update_losses(self, stages, losses):
        """
        Update the losses to those in the internal state
        """
        # if stages not a list turn into a list
        if not isinstance(stages, list):
            stages = [stages]
        contains_last_stage = any(stage.is_last for stage in stages)

        # Return losses if there is a container passed in
        if contains_last_stage and losses is not None:
            if len(self._internal_losses) != self._n_microbatches:
                raise RuntimeError(
                    f"Expecting {self._n_microbatches} losses but got {len(self._internal_losses)}"
                )

            # Clean external container first
            losses.clear()
            # Copy internal losses to external container
            losses.extend(self._internal_losses)

        self._internal_losses.clear()

    def _check_inputs(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Pre-process/check inputs
        """

        def check_type_and_len(mbs, name: str):
            if not isinstance(mbs, list):
                raise TypeError(f"{name} must be a list but got a {type(mbs)}")
            if len(mbs) != self._n_microbatches:
                raise ValueError(
                    f"Expecting {self._n_microbatches} {name} but got {len(mbs)}"
                )

        if arg_mbs is not None:
            check_type_and_len(arg_mbs, "arg_mbs")
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            check_type_and_len(kwarg_mbs, "kwarg_mbs")
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        if target_mbs is not None:
            check_type_and_len(target_mbs, "target_mbs")

        if losses is not None:
            if not isinstance(losses, list):
                raise TypeError(f"losses must be a list but got a {type(losses)}")

        return arg_mbs, kwarg_mbs

    def _compute_loss(self, output, target):
        return self._loss_fn(output, target)  # type: ignore[misc]

    def _merge_outputs(self, output_chunks: List[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """
        return merge_chunks(
            output_chunks,
            self._output_merge_spec,
        )


def _batch_p2p(p2p_ops: List[dist.P2POp], desc: Optional[str] = None):
    """
    Simple wrapper over batch_isend_irecv from torch.distributed, which just adds a descriptive logger on top.
    """
    if len(p2p_ops) == 0:
        return None
    desc_str = f"{desc}, " if desc else ""
    logger.debug("batch_p2p %s%s", desc_str, p2p_ops)
    return dist.batch_isend_irecv(p2p_ops).pop()


def _sorted_batch_p2p(
    p2p_ops: List[dist.P2POp], desc: Optional[str] = None
) -> Dict[int, dist.Work]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # Arrange p2p_ops by peer rank:
    #   int is the peer rank;
    #   List is the list of ops towards the peer
    ops_by_peer: Dict[int, List[dist.P2POp]] = defaultdict(list)
    work_by_peer: Dict[int, dist.Work] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    # Classify the ops by peer rank
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # Call batch_isend_irecv per peer, in sorted order of the peers (to avoid hangs)
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = _batch_p2p(ops, desc=desc)

    return work_by_peer


class PipelineScheduleSingle(_PipelineSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.
    """

    def __init__(
        self,
        stage: _PipelineStageBase,
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        args_chunk_spec: Optional[Tuple[TensorChunkSpec, ...]] = None,
        kwargs_chunk_spec: Optional[Dict[str, TensorChunkSpec]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            args_chunk_spec=args_chunk_spec,
            kwargs_chunk_spec=kwargs_chunk_spec,
            output_merge_spec=output_merge_spec,
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward
        self._stage_initialized = False

    def _initialize_stage(self, args, kwargs):
        self._stage._prepare_forward_infra(self._n_microbatches, args, kwargs)
        if self._has_backward:
            self._stage._prepare_backward_infra(self._n_microbatches)
        self._stage_initialized = True

    def step(self, *args, kwargs_list, input_meta, output_meta, target=None, losses: Optional[List] = None):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        # Clean per iteration
        # before_step = torch.cuda.memory_stats()["active_bytes.all.peak"] / 1024**3
        # if torch.distributed.get_rank() < 2:
        #     print(f"[Rank {torch.distributed.get_rank()}] Before Step Peak: {before_step:.2f} GB")
        self._stage.clear_runtime_states()
        if len(kwargs_list) == 0:
            kwargs_list = [{}] * self._n_microbatches
        self._step_microbatches(args, kwargs_list, input_meta, output_meta, target, losses)
        # Return merged results per original format
        if self._stage.is_last and self._has_backward:
            return None
            # return self._merge_outputs(self._stage.output_chunks)
        else:
            # print(f"Rank {self._stage.device} step ended")
            return None
    
    def set_stage_info(self, input_meta, output_meta):
        # if self._stage.is_last:
            # print(f"Rank {self._stage.device} is last stage")
        # print(f"Stage : {self._stage.stage_index}, Input meta: {input_meta}, Output meta: {[out[0] for out in output_meta]}")
        # print(f"[Rank : {torch.distributed.get_rank()}] starts set stage info")
        # first_block = torch.cuda.memory_stats()["active_bytes.all.peak"] / 1024**3
        for chunk_id in range(self._n_microbatches):
            if not self._stage.is_first:
                recv_infos = tuple(
                    [
                        _RecvInfo(
                            f"recv_for_{self._stage.stage_index}_from_{self._stage.stage_index - 1}",
                            self._stage.stage_index - 1,
                            _make_tensor_from_meta(inp),
                        )
                        for inp in input_meta[chunk_id]
                    ]
                )
                self._stage.args_recv_info[chunk_id] = recv_infos
                 # In case there is backward pass, set requires_grad for receive buffers
                if self._stage.has_backward:
                    for r in recv_infos:
                        r.buffer.requires_grad_(True)

                self._stage.args_recv_info[chunk_id] = recv_infos
            if self._stage.is_first:
                if self._stage.is_vision: # Vision's first stage -> doesn't get inputs from previous stage
                    self._stage.args_recv_info[chunk_id] = tuple(
                        [_RootArgPlaceholder(i) for i in input_meta[chunk_id]]
                    )
                else: # LLM's first stage -> gets inputs from vision stage's connect rank
                    self.vision_connect_rank = dist.get_global_rank(self._stage.vision_last_group, 0)
                    recv_infos = tuple(
                        [
                            _RecvInfo(
                                f"recv_for_{self._stage.stage_index}_from_{self.vision_connect_rank}",
                                self.vision_connect_rank,
                                _make_tensor_from_meta(inp),
                            )
                            for inp in input_meta[chunk_id]
                        ]
                    )
                    self._stage.args_recv_info[chunk_id] = recv_infos
                    # In case there is backward pass, set requires_grad for receive buffers
                    if self._stage.has_backward:
                        for r in recv_infos:
                            r.buffer.requires_grad_(True)

                    self._stage.args_recv_info[chunk_id] = recv_infos
            # print(f"Rank {self._stage.device} args_recv_info for chunk {chunk_id} : {self._stage.args_recv_info[chunk_id]}")
        # Act send info : only contains the next stage index
        # second_block = torch.cuda.memory_stats()["active_bytes.all.peak"] / 1024**3
        for idx in range(len(output_meta[0])):
            if self._stage.is_vision_connect_rank: # Vision's connect rank in vision last stage -> sends outputs to LLM's first stage
                self._stage.act_send_info[idx] = self._stage.llm_first_group_ranks # self._stage.act_send_info will iterate over these ranks
            elif not self._stage.is_last:
                self._stage.act_send_info[idx] = [self._stage.stage_index + 1]
            else: # LLM's last stage -> nothing to send
                self._stage.act_send_info[idx] = []
        # print(f"Rank {self._stage.device} act_send_info : {self._stage.act_send_info}")
        if self._has_backward:
            self.chunks = self._n_microbatches
            for mb_index in range(self._n_microbatches):
                if self._stage.is_vision_connect_rank:
                    out_meta = output_meta[mb_index][0]
                    grad_recv_infos = []
                    for _, dst_list in self._stage.act_send_info.items():
                        grad_recv = []
                        idx = 0
                        for dst in dst_list:
                            if dst in self._stage.llm_tp_first_group_ranks:
                                grad_recv.append(
                                    _RecvInfo(
                                        f"recv_grad_for_{self._stage.stage_index}_from_{dst}",
                                        dst,
                                        _make_tensor_from_meta(
                                            out_meta[idx]
                                        ),
                                    )
                                )
                                idx += 1
                        grad_recv_infos.append(grad_recv)
                    grad_recv_infos = tuple(grad_recv_infos)
                    self._stage.grad_recv_info[mb_index] = grad_recv_infos
                elif not self._stage.is_last: # Make information to recv gradients from stage (in act_send_info)
                    out_meta = output_meta[mb_index]
                    grad_recv_infos = tuple(
                        [
                            _RecvInfo(
                                f"recv_grad_for_{self._stage.stage_index}_from_{dst_list[0]}",
                                dst_list[0],
                                _make_tensor_from_meta(
                                    out_meta[idx]
                                ),
                            )
                            for idx, dst_list in self._stage.act_send_info.items()
                        ]
                    )
                    self._stage.grad_recv_info[mb_index] = grad_recv_infos
                else:
                    self._stage.grad_recv_info[mb_index] = tuple()
                # print(f"Rank {self._stage.device} grad_recv_info for chunk {chunk_id} : {self._stage.grad_recv_info[mb_index]}")
                    # Create bwd send infra lazily
        if self._stage.grad_send_info is None:
            self._stage.grad_send_info = self._stage._create_grad_send_info(self._stage.args_recv_info[0])
class ScheduleGPipe(PipelineScheduleSingle):
    """
    The GPipe schedule.
    Will go through all the microbatches in a fill-drain manner.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        input_meta: Optional[List] = None,
        output_meta: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
        """
        # Delay send waits
        self.set_stage_info(input_meta, output_meta)
        fwd_sends_to_wait: List[dist.Work] = []
        fwd_start = [torch.cuda.Event(enable_timing=True) for _ in range(self._n_microbatches)]
        fwd_end = [torch.cuda.Event(enable_timing=True) for _ in range(self._n_microbatches)]
        bwd_start = [torch.cuda.Event(enable_timing=True) for _ in range(self._n_microbatches)]
        bwd_end = [torch.cuda.Event(enable_timing=True) for _ in range(self._n_microbatches)]

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    work.wait()
                # if self._stage.stage_index == 3:
                #     print(f"Rank {self._stage.device} Forwarding microbatch {i} with args: ", arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]
                fwd_start[i].record()
                output = self._stage.forward_one_chunk(i, arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]
                if not self._stage.is_last:
                    fwd_end[i].record()
                ops = self._stage.get_fwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Forwarded microbatch %s", self._stage.stage_index, i)           
            self._maybe_compute_loss(self._stage, output, target_mbs, i)
            if self._stage.is_last:
                fwd_end[i].record()
        # print(f"Rank {self._stage.device} Forwarded all microbatches")
        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            work.wait()
        # print(f"Rank {self._stage.device} Forward sends finished, has backward: {self._has_backward}")
        # No loss function, no need to run backward
        if not self._has_backward:
            return

        # Run backward
        # Delay send waits
        bwd_sends_to_wait: List[dist.Work] = []
        for i in range(self._n_microbatches):
            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_recv")
                for work in works.values():
                    work.wait()

                bwd_start[i].record()
                loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(
                    i, loss=loss, last_backward=i == self._n_microbatches - 1
                )
                bwd_end[i].record()
                ops = self._stage.get_bwd_send_ops(i)
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())

            logger.debug("[%s] Backwarded microbatch %s", self._stage.stage_index, i)

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()
        # torch.cuda.synchronize()


class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        input_meta: Optional[List] = None,
        output_meta: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """
        # before_m_step = torch.cuda.memory_stats()["active_bytes.all.peak"] / 1024**3
        self.set_stage_info(input_meta, output_meta)
        # after_set_stage_info = torch.cuda.memory_stats()["active_bytes.all.peak"] / 1024**3

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        if self._stage.is_vision:
            warmup_chunks = min(
                self._n_microbatches,
                self._num_stages - self._stage.stage_index,
            )
        else:
            offset = self._num_stages - self._stage.group_size
            warmup_chunks = min(
                self._n_microbatches,
                self._num_stages - (self._stage.stage_index + offset),
            )
        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0

        # Warmup phase
        send_work = None
        fwd_sends = []
        # fwd_start = [torch.cuda.Event(enable_timing=True) for _ in range(self._n_microbatches)]
        # fwd_end = [torch.cuda.Event(enable_timing=True) for _ in range(self._n_microbatches)]
        # bwd_start = [torch.cuda.Event(enable_timing=True) for _ in range(self._n_microbatches)]
        # bwd_end = [torch.cuda.Event(enable_timing=True) for _ in range(self._n_microbatches)]
        # before_m_train = torch.cuda.memory_stats()["active_bytes.all.peak"] / 1024**3
        # if torch.distributed.get_rank() < 2:
        #     print(f"[Rank {torch.distributed.get_rank()}] Before Micro Step Peak: {before_m_step:.2f} GB")
        #     print(f"[Rank {torch.distributed.get_rank()}] After set_stage_info Peak: {after_set_stage_info:.2f} GB")
        #     print(f"[Rank {torch.distributed.get_rank()}] Before Micro Train Peak: {before_m_train:.2f} GB")
        for _ in range(warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)
            if recv_work := _batch_p2p(fwd_recvs, desc="fwd_recv"):
                recv_work.wait()

            # fwd_start[fwd_mb_index].record()
            # Compute
            # torch.cuda.synchronize()
            # start_fwd = time.time()
            output = self._stage.forward_one_chunk(fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]
            # if not (self._stage.is_last and self._stage.is_llm):
                # fwd_end[fwd_mb_index].record()
                # torch.cuda.synchronize()
                # end_fwd = time.time()

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            if send_work:
                send_work.wait()

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            if fwd_mb_index != warmup_chunks - 1:
                # Safe to fire
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            # otherwise:
            #   The last foward send is left for fuse with first 1B in 1B1F below

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)
            # if (self._stage.is_last and self._stage.is_llm):
            #     # fwd_end[fwd_mb_index].record()
            #     torch.cuda.synchronize()
            #     end_fwd = time.time()
            # time_dict["fwd"].append([start_fwd, end_fwd])
            fwd_mb_index += 1
        # Now we should have send ops left over, to be fused with first 1B of 1B1F phase below.
        # print(f"[Rank {self._stage.device.index}] warmup end across all microbatches")
        # 1B1F phase
        while True:  # Don't worry, we have a break inside
            # We actually do 1B first as the `1B1F` name indicates, so prepare its recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)

            # Now, we need to fire the fwd_sends and bwd_recvs together
            if fuse_work := _batch_p2p(fwd_sends + bwd_recvs, desc="fwd_send_bwd_recv"):
                fuse_work.wait()
            # print(f"[Rank {torch.distributed.get_rank()}] recv grad {bwd_mb_index}")
            # Backward one chunk
            # bwd_start[bwd_mb_index].record()
            # torch.cuda.synchronize()
            # start_bwd = time.time()
            # bwd_start = torch.cuda.Event(enable_timing=True)
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )
            # bwd_end[bwd_mb_index].record()
            # torch.cuda.synchronize()
            # end_bwd = time.time()

            # Get the bwd send ops, but don't fire, to be fused with the 1F below
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            bwd_mb_index += 1

            if fwd_mb_index == self._n_microbatches:
                # We are done with 1B1F, so break with some left-over bwd_sends
                break

            # We prepare 1F of the `1B1F`
            fwd_recvs = self._stage.get_fwd_recv_ops(fwd_mb_index)

            # Fuse it with bwd_sends above
            if fuse_work := _batch_p2p(bwd_sends + fwd_recvs, desc="bwd_send_fwd_recv"):
                fuse_work.wait()
            # print(f"[Rank {torch.distributed.get_rank()}] sent grad {bwd_mb_index}")
            # time_dict["bwd"].append([start_bwd, end_bwd])
            # fwd_start[fwd_mb_index].record()
            # Now do the fwd
            # torch.cuda.synchronize()
            # start_fwd = time.time()
            output = self._stage.forward_one_chunk(fwd_mb_index, arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]
            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)
            # fwd_end[fwd_mb_index].record()
            # torch.cuda.synchronize()
            # end_fwd = time.time()
            # time_dict["fwd"].append([start_fwd, end_fwd])
            # Get the fwd send ops, but don't fire, leave it for the next iter (wrap-around)
            fwd_sends = self._stage.get_fwd_send_ops(fwd_mb_index)
            fwd_mb_index += 1

        # Remember we still have some bwd_sends left over after the break? Now it is time to fire it
        send_work = _batch_p2p(bwd_sends, desc="bwd_send")
        # print(f"[Rank {self._stage.device.index}] 1B1F phase end across all microbatches")
        # Cooldown
        while bwd_mb_index < self._n_microbatches:
            # prepare bwd recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops(bwd_mb_index)
            if recv_work := _batch_p2p(bwd_recvs, desc="bwd_recv"):
                recv_work.wait()

            # Backward one chunk
            # bwd_start[bwd_mb_index].record()
            # torch.cuda.synchronize()
            # start_bwd = time.time()
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(
                bwd_mb_index,
                loss=loss,
                last_backward=bwd_mb_index == self._n_microbatches - 1,
            )
            # bwd_end[bwd_mb_index].record()
            # torch.cuda.synchronize()
            # end_bwd = time.time()
            # time_dict["bwd"].append([start_bwd, end_bwd])

            # Clear previous chunk's backward sends (hopefully they have well finished)
            if send_work:
                send_work.wait()

            # Get the bwd send ops, fire it
            bwd_sends = self._stage.get_bwd_send_ops(bwd_mb_index)
            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            bwd_mb_index += 1
        # print(f"[Rank {self._stage.device.index}] Cooldown phase end across all microbatches")
        # Wait for the last backward send to finish
        if send_work:
            send_work.wait()
        # torch.cuda.synchronize()
        # for i in range(self._n_microbatches):
        #     fwd_time = fwd_start[i].elapsed_time(fwd_end[i])
        #     bwd_time = bwd_start[i].elapsed_time(bwd_end[i])
        #     if time_dict is not None:
        #         time_dict["fwd"].append(fwd_time)
        #         time_dict["bwd"].append(bwd_time)
        # Return losses if there is a container passed in
        # print(f"[Rank {self._stage.device.index}] : fwd time : {fwd_time}, bwd time: {bwd_time}")
        self._update_losses(self._stage, losses)
