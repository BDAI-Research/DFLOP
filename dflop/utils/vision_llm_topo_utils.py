import os
import torch
import numpy as np
import torch.distributed as dist
from typing import Dict, Any

# ==============================================================================
# User-configurable settings (same as before)
# The logical structure of components is independent of their physical node location.
# ==============================================================================

def set_topology_config(world_size, vision_ranks, vision_pp_size, vision_dp_size, vision_tp_size,
                        llm_pp_size, llm_dp_size, llm_tp_size):
    """Dynamically generate topology configuration"""
    return {
        "vision": {
            "ranks": range(vision_ranks),
            "pp_size": vision_pp_size,
            "dp_size": vision_dp_size,
            "tp_size": vision_tp_size,
        },
        "llm": {
            "ranks": range(vision_ranks, world_size),
            "pp_size": llm_pp_size,
            "dp_size": llm_dp_size,
            "tp_size": llm_tp_size,
        }
    }

# ==============================================================================

def _validate_topology(config: Dict[str, Any], world_size: int):
    """Validate that the provided topology configuration is correct."""
    all_ranks_in_config = set()
    for name, component_config in config.items():
        num_ranks = len(component_config["ranks"])
        expected_ranks = component_config["pp_size"] * component_config["dp_size"] * component_config["tp_size"]
        if num_ranks != expected_ranks:
            raise ValueError(f"'{name}' component configuration error: pp*dp*tp ({expected_ranks}) does not match the assigned number of ranks ({num_ranks}).")
        component_ranks = set(component_config["ranks"])
        if not component_ranks.isdisjoint(all_ranks_in_config):
            raise ValueError("Configuration error: There are duplicate ranks between components.")
        all_ranks_in_config.update(component_ranks)
    expected_all_ranks = set(range(world_size))
    if all_ranks_in_config != expected_all_ranks:
        raise ValueError(f"Configuration error: Configured ranks ({all_ranks_in_config}) do not match actual world_size ({world_size}).")
    print("Topology configuration is valid.")


def setup_multinode_distributed_groups(topology_config: Dict[str, Any]):
    """
    Create parallel processing groups based on flexible topology configuration for multi-node environments.
    """
    # 1. Initialize basic distributed environment
    # torchrun sets environment variables like RANK, WORLD_SIZE, and LOCAL_RANK.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])  # <-- Get local rank

    if rank == 0:
        print(f"A total of {world_size} ranks will be set up in the multi-node environment.")
        _validate_topology(topology_config, world_size)

    # Each rank determines its role based on the global rank.
    # This logic is independent of the node location and is the same as before.
    rank_details = [{} for _ in range(world_size)]
    pp_group_ranks, dp_group_ranks, tp_group_ranks = [], [], []
    for name, config in topology_config.items():
        pp, dp, tp = config["pp_size"], config["dp_size"], config["tp_size"]
        module_ranks = np.arange(config["ranks"][0], config["ranks"][-1] + 1).reshape(pp, dp, tp)
        if rank in config["ranks"]:
            cur_module_ranks = module_ranks
        for p in range(pp):
            for d in range(dp):
                tp_group_ranks.append(module_ranks[p, d, :].tolist())
        for t in range(tp):
            for p in range(pp):
                dp_group_ranks.append(module_ranks[p, :, t].tolist())

        for t in range(tp):
            for d in range(dp):
                pp_group_ranks.append(module_ranks[:, d, t].tolist())

        # tp_group_ranks.append(tp_ranks)
        # dp_group_ranks.append(dp_ranks)
        # pp_group_ranks.append(pp_ranks)
        offset = config["ranks"][0]
        # Calculate tp, dp, pp rank for each component
        for r in config["ranks"]:
            local_r = r - offset
            rank_details[r] = {
                "component": name,
                "local_rank_in_comp": local_r,
                "pp_rank": (local_r // (tp * dp)) % pp,
                "dp_rank": (local_r // tp) % dp,
                "tp_rank": local_r % tp,
            }
    for ranks_in_tp_group in tp_group_ranks:
        # print(f"TP : {ranks_in_tp_group}")
        group = dist.new_group(ranks_in_tp_group)
        if rank in ranks_in_tp_group:
            module_tp_group = group
    for ranks_in_dp_group in dp_group_ranks:
        # print(f"DP : {ranks_in_dp_group}")
        group = dist.new_group(ranks_in_dp_group)
        if rank in ranks_in_dp_group:
            module_dp_group = group
    for ranks_in_pp_group in pp_group_ranks:
        group = dist.new_group(ranks_in_pp_group)
        if rank in ranks_in_pp_group:
            module_pp_group = group
    cur_rank_info = rank_details[rank]
    cur_component = cur_rank_info["component"]

    v_pp_size = topology_config["vision"]["pp_size"]
    v_pp_last_group_ranks = []
    l_pp_first_group_ranks = []
    v_tp_first_group_ranks = []
    l_tp_first_group_ranks = []


    for i in range(world_size):
        i_details = rank_details[i]
        # i_dp_rank = i_details["dp_rank"]
        i_tp_rank = i_details["tp_rank"]
        i_pp_rank = i_details["pp_rank"]
        # Create dist groups per component
        # if i_details["component"] == cur_component:  # Each parallel group must have the same parallel rank in the other dimensions.
        #     # if i_dp_rank == cur_rank_info["dp_rank"] and i_tp_rank == cur_rank_info["tp_rank"]:
        #     #     pp_group_ranks.append(i)
        #     # if i_pp_rank == cur_rank_info["pp_rank"] and i_tp_rank == cur_rank_info["tp_rank"]:
        #     #     dp_group_ranks.append(i)
        #     # if i_pp_rank == cur_rank_info["pp_rank"] and i_dp_rank == cur_rank_info["dp_rank"]:
        #     #     tp_group_ranks.append(i)
        if (i_details["component"] == "vision") & (i_pp_rank == v_pp_size - 1):
            v_pp_last_group_ranks.append(i)
            if i_tp_rank == 0:
                v_tp_first_group_ranks.append(i)
        if (i_details["component"] == "llm") & (i_pp_rank == 0):
            l_pp_first_group_ranks.append(i)
            if i_tp_rank == 0:
                l_tp_first_group_ranks.append(i)

    # v_pp_last_group = dist.new_group(v_pp_last_group_ranks)
    # l_pp_first_group = dist.new_group(l_pp_first_group_ranks)
    print(f"Vision PP Last Group Ranks: {v_pp_last_group_ranks}, LLM PP First Group Ranks: {l_pp_first_group_ranks}")
    
    # Wait until all processes have finished creating their groups
    dist.barrier()
    
    if rank == 0:
        print(f"TP Group Ranks : {tp_group_ranks}")
        print(f"DP Group Ranks : {dp_group_ranks}")
        print(f"PP Group Ranks : {pp_group_ranks}")
        print("\nAll process groups have been successfully created.\n")
        
    return {
        "pp_group": module_pp_group,
        "dp_group": module_dp_group,
        "tp_group": module_tp_group,
        "v_pp_last_group_ranks": v_pp_last_group_ranks,
        "l_pp_first_group_ranks": l_pp_first_group_ranks,
        "v_tp_first_group_ranks": v_tp_first_group_ranks,
        "l_tp_first_group_ranks": l_tp_first_group_ranks,
        "pp_size": {"vision" : topology_config["vision"]["pp_size"], 
                  "llm" : topology_config["llm"]["pp_size"]},
        "offset" : 0 if cur_component == "vision" else topology_config["vision"]["pp_size"],
        "component": cur_component,
        "module_ranks" : cur_module_ranks,
        "details": cur_rank_info
    }
