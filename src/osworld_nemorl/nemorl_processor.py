"""Custom NeMo-RL VLM processor for OSWorld trajectory samples."""

from __future__ import annotations

import os
from typing import Any


def _normalize_user_content(user_content: Any) -> list[dict[str, Any]]:
    if isinstance(user_content, str):
        return [{"type": "text", "text": user_content}]
    if isinstance(user_content, list):
        normalized: list[dict[str, Any]] = []
        for part in user_content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                normalized.append({"type": "text", "text": str(part.get("text", ""))})
            elif part_type == "image":
                image_ref = part.get("image")
                if image_ref:
                    normalized.append({"type": "image", "image": image_ref})
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                if isinstance(image_url, dict) and image_url.get("url"):
                    normalized.append({"type": "image", "image": image_url["url"]})
        return normalized
    return [{"type": "text", "text": str(user_content)}]


def _extract_assistant_text(assistant_content: Any) -> str:
    if isinstance(assistant_content, str):
        return assistant_content
    if isinstance(assistant_content, list):
        text_parts: list[str] = []
        for part in assistant_content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        return "\n".join(p for p in text_parts if p).strip()
    if assistant_content is None:
        return ""
    return str(assistant_content)


def _parse_positive_int(value: str | None) -> int:
    if value is None:
        return 0
    value = str(value).strip()
    if not value:
        return 0
    try:
        parsed = int(value)
    except ValueError:
        return 0
    return parsed if parsed > 0 else 0


def _downscale_image_if_needed(image: Any, max_side: int) -> Any:
    if max_side <= 0 or not hasattr(image, "size"):
        return image

    try:
        width, height = image.size
    except Exception:
        return image

    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )

    try:
        from PIL import Image as PILImage

        resampling_attr = getattr(PILImage, "Resampling", PILImage)
        resample = getattr(resampling_attr, "LANCZOS", PILImage.BICUBIC)
        return image.resize(new_size, resample=resample)
    except Exception:
        try:
            return image.resize(new_size)
        except Exception:
            return image


def osworld_vlm_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: Any,
    processor: Any,
    max_seq_length: int,
    idx: int,
) -> dict[str, Any]:
    """Process converted OSWorld samples into NeMo-RL VLM DatumSpec."""
    from nemo_rl.data.multimodal_utils import (
        PackedTensor,
        get_dim_to_pack_along,
        get_multimodal_keys_from_processor,
        resolve_to_image,
    )

    messages = datum_dict.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("OSWorld sample must contain at least [user, assistant] messages.")

    user_raw = messages[0]
    assistant_raw = messages[1]
    if not isinstance(user_raw, dict) or not isinstance(assistant_raw, dict):
        raise ValueError("messages entries must be dicts")

    user_parts = _normalize_user_content(user_raw.get("content", ""))
    ground_truth = _extract_assistant_text(assistant_raw.get("content", ""))

    user_message: dict[str, Any] = {"role": "user", "content": []}
    images = []
    for part in user_parts:
        if part["type"] == "text":
            text = str(part.get("text", ""))
            if task_data_spec.prompt:
                text = task_data_spec.prompt.format(text)
            user_message["content"].append({"type": "text", "text": text})
        elif part["type"] == "image":
            image_value = part.get("image")
            if image_value is None:
                continue
            user_message["content"].append({"type": "image", "image": image_value})
            images.append(image_value)

    if len(user_message["content"]) == 0:
        user_message["content"] = [{"type": "text", "text": ""}]

    images = [resolve_to_image(image) for image in images]
    max_image_side = _parse_positive_int(os.environ.get("OSWORLD_VLM_MAX_IMAGE_SIDE"))
    if max_image_side > 0:
        images = [_downscale_image_if_needed(image, max_image_side) for image in images]

    if hasattr(processor, "conversation_preprocessor"):
        user_message_for_chat_template = processor.conversation_preprocessor(user_message)
    else:
        user_message_for_chat_template = user_message

    string_formatted_dialog = processor.apply_chat_template(
        [user_message_for_chat_template],
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = processor.apply_chat_template(
        [user_message],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    user_message["token_ids"] = model_inputs["input_ids"][0]
    multimodal_keys = get_multimodal_keys_from_processor(processor)
    for key in multimodal_keys:
        if key in model_inputs:
            user_message[key] = PackedTensor(
                model_inputs[key], dim_to_pack=get_dim_to_pack_along(processor, key)
            )

    if "token_type_ids" in model_inputs:
        user_message["token_type_ids"] = model_inputs["token_type_ids"][0]
    if "mm_token_type_ids" in model_inputs:
        user_message["mm_token_type_ids"] = model_inputs["mm_token_type_ids"][0]

    message_log = [user_message]
    length = sum(len(m["token_ids"]) for m in message_log)
    loss_multiplier = 1.0

    if length >= max_seq_length:
        vllm_kwargs = {
            "vllm_content": None,
            "vllm_images": [],
            "vllm_audios": [],
        }
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
            for key, value in list(chat_message.items()):
                if isinstance(value, PackedTensor):
                    chat_message[key] = PackedTensor.empty_like(value)
        loss_multiplier = 0.0
    else:
        vllm_kwargs = {
            "vllm_content": string_formatted_dialog,
            "vllm_images": images,
            "vllm_audios": [],
        }

    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": {"ground_truth": ground_truth},
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict.get("task_name", "osworld_action_imitation"),
        **vllm_kwargs,
    }
