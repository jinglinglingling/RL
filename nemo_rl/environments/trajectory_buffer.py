# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Ordered, token-faithful storage for interactive agent trajectories."""

from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


@dataclass(frozen=True)
class TrajectoryTurn:
    """One policy decision and the exact context used to generate it."""

    message_log: list[dict[str, Any]]
    prompt_token_ids: list[int]
    generation_token_ids: list[int]
    multimodal_inputs: dict[str, Any]

    @property
    def context_after_action(self) -> list[int]:
        return self.prompt_token_ids + self.generation_token_ids

    @property
    def images(self) -> list[str]:
        return list(self.multimodal_inputs.get("images_base64") or [])


@dataclass(frozen=True)
class TrajectoryTrainingUnit:
    """A prefix-monotonic trajectory segment that can be trained jointly."""

    message_log: list[dict[str, Any]]
    multimodal_inputs: dict[str, Any]
    turn_indices: tuple[int, ...]

    @property
    def turn_count(self) -> int:
        return len(self.turn_indices)


@dataclass
class TrajectoryBuffer:
    """Append-only rollout record used to build full-trajectory loss inputs.

    A later turn can share a training unit with earlier turns only when both
    its token context and image history extend the previous context exactly.
    Compaction may rewrite either prefix; in that case a new training unit is
    started instead of constructing an off-policy synthetic sequence.
    """

    turns: list[TrajectoryTurn] = field(default_factory=list)

    def append(self, turn: TrajectoryTurn) -> None:
        if not turn.message_log:
            raise ValueError("A trajectory turn must contain a message log.")
        self.turns.append(turn)

    def __len__(self) -> int:
        return len(self.turns)

    def __iter__(self) -> Iterator[TrajectoryTurn]:
        return iter(self.turns)

    def __getitem__(self, index: int) -> TrajectoryTurn:
        return self.turns[index]

    def build_monotonic_training_units(
        self,
    ) -> tuple[list[TrajectoryTrainingUnit], list[int]]:
        """Return maximal full-trajectory segments and turn-to-unit mapping."""

        if not self.turns:
            raise ValueError("Cannot build training units from an empty trajectory.")

        units: list[TrajectoryTrainingUnit] = []
        turn_to_unit: list[int] = []
        current_message_log: list[dict[str, Any]] | None = None
        current_context_ids: list[int] = []
        current_images: list[str] = []
        current_multimodal_inputs: dict[str, Any] | None = None
        current_turn_indices: list[int] = []

        def finalize() -> None:
            nonlocal current_message_log
            if current_message_log is None or current_multimodal_inputs is None:
                return
            units.append(
                TrajectoryTrainingUnit(
                    message_log=current_message_log,
                    multimodal_inputs=current_multimodal_inputs,
                    turn_indices=tuple(current_turn_indices),
                )
            )
            current_message_log = None

        for turn_idx, turn in enumerate(self.turns):
            token_prefix_matches = (
                current_message_log is not None
                and len(turn.prompt_token_ids) >= len(current_context_ids)
                and turn.prompt_token_ids[: len(current_context_ids)]
                == current_context_ids
            )
            image_prefix_matches = (
                len(turn.images) >= len(current_images)
                and turn.images[: len(current_images)] == current_images
            )

            if not (token_prefix_matches and image_prefix_matches):
                finalize()
                current_message_log = [dict(message) for message in turn.message_log]
                current_turn_indices = [turn_idx]
            else:
                delta_start = len(current_context_ids)
                full_user_message = turn.message_log[0]
                delta_user_message: dict[str, Any] = {
                    "role": "user",
                    "content": "",
                    "token_ids": torch.tensor(
                        turn.prompt_token_ids[delta_start:], dtype=torch.long
                    ),
                }
                if "routed_experts" in full_user_message:
                    delta_user_message["routed_experts"] = full_user_message[
                        "routed_experts"
                    ][delta_start:]
                current_message_log.extend(
                    [delta_user_message, dict(turn.message_log[1])]
                )
                current_turn_indices.append(turn_idx)

            turn_to_unit.append(len(units))
            current_context_ids = turn.context_after_action
            current_images = turn.images
            current_multimodal_inputs = turn.multimodal_inputs

        finalize()
        return units, turn_to_unit
