# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 GLM-5.3 small-N tensor-core GEMV: FP64 oracle, FP32 output, graph
replay, determinism, shape gates, and the GLM shared-MLP dispatch."""

import socket

import pytest
import torch

import vllm._sm70_ops as sm70_ops
from vllm.platforms import current_platform

requires_sm70 = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.get_device_capability() == (7, 0)
        and hasattr(torch.ops._C, "sm70_glm53_small_n_gemv_out")
    ),
    reason="requires V100 and the SM70 small-N GEMV op",
)

SHAPES = [(288, 4096), (512, 4096), (1024, 4096), (4096, 256), (4096, 2048)]


def _inputs(num_tokens, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((num_tokens, k), device="cuda", generator=g).half()
    w = (torch.randn((n, k), device="cuda", generator=g) * 0.02).half()
    return x, w


@requires_sm70
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("num_tokens", [1, 3, 4, 8])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
def test_small_n_gemv_matches_fp64(n, k, num_tokens, out_dtype):
    x, w = _inputs(num_tokens, n, k, seed=n + num_tokens)
    y = torch.empty((num_tokens, n), device="cuda", dtype=out_dtype)
    sm70_ops.sm70_glm53_small_n_gemv_out(y, x, w)
    torch.cuda.synchronize()
    ref = x.double() @ w.double().T
    if out_dtype == torch.float32:
        torch.testing.assert_close(y.double(), ref, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(y.double(), ref, rtol=2e-3, atol=2e-3)
    # mutation: a different weight must not reproduce the output
    y2 = torch.empty_like(y)
    sm70_ops.sm70_glm53_small_n_gemv_out(y2, x, torch.roll(w, 1, dims=0).contiguous())
    assert not torch.equal(y, y2)


@requires_sm70
def test_small_n_gemv_fp32_output_is_the_accumulator():
    """FP32 output skips the FP16 rounding the FP16 output has."""
    x, w = _inputs(4, 288, 4096, seed=7)
    y16 = torch.empty((4, 288), device="cuda", dtype=torch.float16)
    y32 = torch.empty((4, 288), device="cuda", dtype=torch.float32)
    sm70_ops.sm70_glm53_small_n_gemv_out(y16, x, w)
    sm70_ops.sm70_glm53_small_n_gemv_out(y32, x, w)
    torch.cuda.synchronize()
    torch.testing.assert_close(y32.half(), y16, rtol=0, atol=0)
    assert not torch.equal(y32, y16.float())


@requires_sm70
def test_small_n_gemv_graph_replay_and_determinism():
    x, w = _inputs(4, 512, 4096, seed=3)
    y = torch.empty((4, 512), device="cuda", dtype=torch.float16)
    sm70_ops.sm70_glm53_small_n_gemv_out(y, x, w)
    torch.cuda.synchronize()
    eager = y.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        sm70_ops.sm70_glm53_small_n_gemv_out(y, x, w)
    for _ in range(3):
        y.zero_()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(y, eager, rtol=0, atol=0)


@requires_sm70
@pytest.mark.parametrize(
    "num_tokens,n,k", [(9, 288, 4096), (4, 300, 4096), (4, 288, 4000)]
)
def test_small_n_gemv_rejects_unsupported_shapes(num_tokens, n, k):
    x, w = _inputs(num_tokens, n, k)
    y = torch.empty((num_tokens, n), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError):
        sm70_ops.sm70_glm53_small_n_gemv_out(y, x, w)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@requires_sm70
def test_glm_shared_mlp_dispatch(monkeypatch):
    """With the policy unset the GLM shared MLP takes the native op for 1-8
    tokens and matches the standard path; an explicit False keeps cuBLAS."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.models.glm5next.nvidia import model as glm_model

    calls: list[tuple[int, int]] = []
    real = sm70_ops.sm70_glm53_small_n_gemv_out

    def spy(y, x, w):
        calls.append(tuple(w.shape))
        return real(y, x, w)

    monkeypatch.setattr(sm70_ops, "sm70_glm53_small_n_gemv_out", spy)

    def patch_policy(value, source):
        real_policy = glm_model.layer_policy()

        class _Policy:
            glm_small_n_gemv = value
            sources = {
                **dict(getattr(real_policy, "sources", {})),
                "glm_small_n_gemv": source,
            }

            def __getattr__(self, name):
                return getattr(real_policy, name)

        policy = _Policy()
        monkeypatch.setattr(glm_model, "layer_policy", lambda cfg=None: policy)

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(1, 0, f"tcp://127.0.0.1:{_free_port()}", 0, "nccl")
        initialize_model_parallel(1, 1)
        try:
            torch.manual_seed(0)
            mlp = glm_model.Glm5NextMLP(
                4096, 256, "silu", reduce_results=False, prefix="t"
            )
            mlp = mlp.cuda().half()
            with torch.no_grad():
                for p in mlp.parameters():
                    p.copy_(torch.randn_like(p) * 0.02)
            x = torch.randn((4, 4096), device="cuda").half()

            patch_policy(None, "default")
            calls.clear()
            native = mlp(x)
            assert calls == [(512, 4096), (4096, 256)], calls

            patch_policy(False, "typed")
            calls.clear()
            standard = mlp(x)
            assert calls == []

            torch.testing.assert_close(native, standard, rtol=2e-2, atol=2e-3)
            x64 = x.double()
            gu = x64 @ mlp.gate_up_proj.weight.double().T
            ref = (
                torch.nn.functional.silu(gu[:, :256]) * gu[:, 256:]
            ) @ mlp.down_proj.weight.double().T
            torch.testing.assert_close(native.double(), ref, rtol=2e-2, atol=2e-3)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
