import torch

from nemo_rl.environments.trajectory_buffer import (
    TrajectoryBuffer,
    TrajectoryTurn,
)


def _turn(
    prompt: list[int],
    generation: list[int],
    images: list[str],
) -> TrajectoryTurn:
    return TrajectoryTurn(
        message_log=[
            {
                "role": "user",
                "content": "",
                "token_ids": torch.tensor(prompt),
            },
            {
                "role": "assistant",
                "content": "",
                "token_ids": torch.tensor(generation),
            },
        ],
        prompt_token_ids=prompt,
        generation_token_ids=generation,
        multimodal_inputs={"images_base64": images},
    )


def test_trajectory_buffer_builds_one_full_monotonic_unit() -> None:
    buffer = TrajectoryBuffer()
    buffer.append(_turn([1], [10], ["a"]))
    buffer.append(_turn([1, 10, 2], [20], ["a", "b"]))
    buffer.append(_turn([1, 10, 2, 20, 3], [30], ["a", "b", "c"]))

    units, turn_to_unit = buffer.build_monotonic_training_units()

    assert turn_to_unit == [0, 0, 0]
    assert len(units) == 1
    assert units[0].turn_indices == (0, 1, 2)
    assert [message["token_ids"].tolist() for message in units[0].message_log] == [
        [1],
        [10],
        [2],
        [20],
        [3],
        [30],
    ]
    assert units[0].multimodal_inputs["images_base64"] == ["a", "b", "c"]


def test_trajectory_buffer_splits_when_compaction_rewrites_prefix() -> None:
    buffer = TrajectoryBuffer()
    buffer.append(_turn([1], [10], ["a"]))
    buffer.append(_turn([1, 10, 2], [20], ["a", "b"]))
    buffer.append(_turn([99, 2, 20, 3], [30], ["b", "c"]))

    units, turn_to_unit = buffer.build_monotonic_training_units()

    assert turn_to_unit == [0, 0, 1]
    assert [unit.turn_indices for unit in units] == [(0, 1), (2,)]
    assert [unit.turn_count for unit in units] == [2, 1]
