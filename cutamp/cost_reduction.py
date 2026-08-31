# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Dict, Set, Optional

import torch
from jaxtyping import Float


def cost_breakdown(cost_dict: Dict[str, dict], idx: int, cost_reducer: "CostReducer") -> dict:
    """Per-term cost values for a single particle, for logging.

    Mirrors CostReducer.get_cost so recorded numbers match the objective: each entry sums over the
    time dimension and reports raw value, the applied weight (an ABSENT multiplier is 1.0, as in the
    reducer), and their product. ``kind`` is "cost" (soft) or "constraint" (hard). Keyed
    "<CostType>/<name>", e.g. "GraspCost/grasp_rot_change", "TrajectoryLength/traj_length".

    Module-level rather than a ParticleOptimizer method because the optimizer is not the only thing
    that needs it: the particle that gets EXECUTED is chosen after optimization, by
    ``get_ranked_satisfying_particles``, and is a different particle whenever that ranking is not
    soft-cost ordered.
    """
    breakdown = {}
    for cost_type, entry in cost_dict.items():
        for name, values in entry["values"].items():
            v = values[idx]
            if v.ndim >= 1:
                v = v.sum()  # sum over time, matching the reducer
            raw = v.item()
            mult = cost_reducer.cost_to_multiplier.get((cost_type, name))
            weight = 1.0 if mult is None else float(mult)
            breakdown[f"{cost_type}/{name}"] = {
                "raw": raw,
                "weight": weight,
                "weighted": raw * weight,
                "kind": entry["type"],
            }
    return breakdown


class CostReducer:
    """Reduces the cost dictionary to a single cost per particle by applying a weighted sum of costs."""

    def __init__(self, cost_config: Dict[str, Dict[str, float]]):
        self.cost_config = cost_config
        # Flatten the nested config for fast lookup
        self.cost_to_multiplier = {
            (cost_type, name): multiplier
            for cost_type, costs in cost_config.items()
            for name, multiplier in costs.items()
        }

    def _get_multiplier(self, cost_type: str, name: str) -> Optional[float]:
        return self.cost_to_multiplier.get((cost_type, name))

    def get_cost(self, cost_dict: Dict[str, dict], consider_types: Set[str]) -> Float[torch.Tensor, "num_particles"]:
        """Returns total cost per particle by taking weighted sum of considered cost types."""
        cost = None
        for cost_type, entry in cost_dict.items():
            if entry["type"] not in consider_types:
                continue

            for name, values in entry["values"].items():
                if values.ndim == 2:
                    values = values.sum(dim=1)  # Sum over time
                multiplier = self._get_multiplier(cost_type, name)
                if multiplier is not None:
                    values = values * multiplier
                cost = values if cost is None else cost + values
        return cost

    def soft_costs(self, cost_dict: Dict[str, dict]) -> Float[torch.Tensor, "num_particles"]:
        """Reduce only the soft costs."""
        return self.get_cost(cost_dict, consider_types={"cost"})

    def hard_costs(self, cost_dict: Dict[str, dict]) -> Float[torch.Tensor, "num_particles"]:
        """Reduce only the constraints === hard costs."""
        return self.get_cost(cost_dict, consider_types={"constraint"})

    def __call__(
        self, cost_dict: Dict[str, dict], consider_types: Set[str] = frozenset(("constraint", "cost"))
    ) -> Float[torch.Tensor, "num_particles"]:
        """Sum both soft and hard costs."""
        return self.get_cost(cost_dict, consider_types=consider_types)
