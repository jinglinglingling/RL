# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""prompt_mean loss normalization on the split-API (SingleController) trainer path."""

import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import MetricNormalizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _config(loss_normalization: str, **overrides) -> ClippedPGLossConfig:
    settings = dict(
        reference_policy_kl_penalty=0.0,
        ratio_clip_min=0.2,
        ratio_clip_max=0.2,
        ratio_clip_c=None,
        use_on_policy_kl_approximation=False,
        use_importance_sampling_correction=False,
        token_level_loss=True,
        loss_normalization=loss_normalization,
    )
    return ClippedPGLossConfig(**(settings | overrides))


def _batch(rows: int = 4, seq_len: int = 6, vocab: int = 16):
    """Two comparison groups of two rows, two eligible action tokens per row."""
    torch.manual_seed(0)
    token_mask = torch.zeros(rows, seq_len)
    token_mask[:, -2:] = 1.0
    data = BatchedDataDict(
        {
            "input_ids": torch.randint(0, vocab, (rows, seq_len)),
            "token_mask": token_mask,
            "sample_mask": torch.ones(rows),
            "advantages": torch.tensor([1.0, -1.0, 1.0, -1.0])
            .unsqueeze(-1)
            .expand(rows, seq_len)
            .clone(),
            "prev_logprobs": torch.randn(rows, seq_len) * 0.1 - 2.0,
            "generation_logprobs": torch.randn(rows, seq_len) * 0.1 - 2.0,
            "reference_policy_logprobs": torch.zeros(rows, seq_len),
            # 2 groups x (2 rows x 2 tokens): w_row = 1 / (G * tokens_in_group)
            "loss_row_weight": torch.full((rows,), 1.0 / (2 * 4)),
        }
    )
    logits = torch.randn(rows, seq_len, vocab)
    return data, logits


def _loss_and_grad(loss_fn, data, logits, *, global_valid_seqs, global_valid_toks):
    logits = logits.clone().requires_grad_(True)
    # CPU stand-in for prepare_loss_input (whose logits->logprobs helper is CUDA-only).
    next_token_logprobs = torch.gather(
        torch.log_softmax(logits[:, :-1], dim=-1),
        dim=-1,
        index=data["input_ids"][:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    loss, metrics = loss_fn(
        next_token_logprobs=next_token_logprobs,
        data=data,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )
    loss.backward()
    return loss.detach(), logits.grad.detach().clone(), metrics


def test_prompt_mean_advertises_unit_gradient_denominator():
    loss_fn = ClippedPGLossFn(_config("prompt_mean"))
    assert loss_fn.gradient_normalization is MetricNormalizer.NONE
    assert loss_fn.metric_normalizations["loss"] is MetricNormalizer.NONE
    token_mean = ClippedPGLossFn(_config("global_action_token_mean"))
    assert token_mean.gradient_normalization is MetricNormalizer.TOKENS
    assert token_mean.metric_normalizations["loss"] is MetricNormalizer.TOKENS


def test_prompt_mean_is_invariant_to_split_path_placeholder_denominators():
    """Split-API trainers call the loss with global_valid_*=1 and rescale by 1/N
    afterwards; prompt_mean must return the same (already normalized) loss and
    gradient either way, and equal token-mean when groups have equal token counts.
    """
    data, logits = _batch()
    n_toks = torch.tensor(8.0)
    n_seqs = torch.tensor(4.0)
    one = torch.tensor(1.0)

    prompt_mean = ClippedPGLossFn(_config("prompt_mean"))
    loss_true, grad_true, _ = _loss_and_grad(
        prompt_mean, data, logits, global_valid_seqs=n_seqs, global_valid_toks=n_toks
    )
    loss_placeholder, grad_placeholder, _ = _loss_and_grad(
        prompt_mean, data, logits, global_valid_seqs=one, global_valid_toks=one
    )
    torch.testing.assert_close(loss_placeholder, loss_true)
    torch.testing.assert_close(grad_placeholder, grad_true)

    token_mean = ClippedPGLossFn(_config("global_action_token_mean"))
    loss_tm, grad_tm, _ = _loss_and_grad(
        token_mean, data, logits, global_valid_seqs=n_seqs, global_valid_toks=n_toks
    )
    torch.testing.assert_close(loss_true, loss_tm)
    torch.testing.assert_close(grad_true, grad_tm)
    # Token-mean under the placeholder is the raw sum (N times larger): the
    # split path's 1/N rescale is what prompt_mean must NOT receive.
    loss_tm_raw, _, _ = _loss_and_grad(
        token_mean, data, logits, global_valid_seqs=one, global_valid_toks=one
    )
    torch.testing.assert_close(loss_tm_raw, loss_tm * n_toks)


def test_prompt_mean_requires_materialized_row_weights_and_no_kl():
    data, logits = _batch()
    del data["loss_row_weight"]
    loss_fn = ClippedPGLossFn(_config("prompt_mean"))
    with pytest.raises(KeyError, match="loss_row_weight"):
        _loss_and_grad(
            loss_fn,
            data,
            logits,
            global_valid_seqs=torch.tensor(4.0),
            global_valid_toks=torch.tensor(8.0),
        )
    with pytest.raises(ValueError, match="reference_policy_kl_penalty"):
        ClippedPGLossFn(_config("prompt_mean", reference_policy_kl_penalty=0.01))
    with pytest.raises(ValueError, match="token_level_loss"):
        ClippedPGLossFn(_config("prompt_mean", token_level_loss=False))
