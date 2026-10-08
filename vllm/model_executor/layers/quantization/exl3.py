# SPDX-License-Identifier: Apache-2.0
"""EXL3 (trellis-coded) routed experts for GLM-5.3-Flash on exact SM70 (V100).

Checkpoint contract (produced by repack_1cat.py from an exllamav3 conversion with
EXL3_TILE_ORDER=sm70_colmajor): only the routed MoE experts are EXL3 (K=2, mcg codebook,
column-major tile order); every other tensor is plain FP16/BF16 and takes the existing
SM70 GLM-5.3 paths. Per MoE layer the experts are stored stacked:
  experts.exl3_{gate,up}_trellis [E, H/16, I/16, 16K] int16, _suh [E, H], _svh [E, I]
  experts.exl3_down_trellis      [E, I/16, H/16, 16K] int16, _suh [E, I], _svh [E, H]
K (2, 3 or 4 bits) may differ per layer: quantization_config.layer_bits = {layer: K}, default .bits.
Tensor parallelism slices the intermediate dimension (trellis tile columns of gate/up,
tile rows of down); 512-wide slices keep the 128-point Hadamards rank-local.
"""
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

logger = init_logger(__name__)

_MAX_TOKENS_PER_LAUNCH = 1024


class Exl3Config(QuantizationConfig):
    def __init__(
        self,
        bits: int = 2,
        codebook: str = "mcg",
        tile_order: str = "sm70_colmajor",
        layer_bits: dict[str, int] | None = None,
    ):
        super().__init__()
        self.layer_bits = {int(k): int(v) for k, v in (layer_bits or {}).items()}
        if (
            bits not in (2, 3, 4)
            or any(v not in (2, 3, 4) for v in self.layer_bits.values())
            or codebook != "mcg"
            or tile_order != "sm70_colmajor"
        ):
            raise ValueError(
                "The SM70 EXL3 path supports K=2/3/4 per layer, mcg codebook, sm70_colmajor tile order; "
                f"got bits={bits}, layer_bits={self.layer_bits}, codebook={codebook}, tile_order={tile_order}."
            )
        self.bits, self.codebook, self.tile_order = bits, codebook, tile_order

    def bits_for(self, prefix: str) -> int:
        """Routed-expert bit width of the MoE layer named by a module prefix (...layers.<N>.mlp.experts)."""
        parts = prefix.split(".")
        for i, part in enumerate(parts[:-1]):
            if part == "layers" and parts[i + 1].isdigit():
                return self.layer_bits.get(int(parts[i + 1]), self.bits)
        return self.bits

    def __repr__(self) -> str:
        return f"Exl3Config(bits={self.bits}, codebook={self.codebook}, tile_order={self.tile_order})"

    @classmethod
    def get_name(cls):
        return "exl3"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(
            bits=int(config.get("bits", 2)),
            codebook=config.get("codebook", "mcg"),
            tile_order=config.get("tile_order", "sm80"),
            layer_bits=config.get("layer_bits"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            return Exl3SM70MoEMethod(layer.moe_config, self.bits_for(prefix))
        return None


class Exl3SM70MoEMethod(FusedMoEMethodBase):
    """Routed experts decoded on the fly from the EXL3 trellis by native SM70 kernels."""

    def __init__(self, moe: FusedMoEConfig, bits: int = 2):
        super().__init__(moe)
        self.bits = bits

    @property
    def supports_eplb(self) -> bool:
        return False

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        E, H, I = num_experts, hidden_size, intermediate_size_per_partition
        if H % 128 or I % 128:
            raise ValueError(f"EXL3 SM70 MoE needs 128-divisible dims per rank, got H={H}, I={I}.")
        rank = layer.tp_rank
        layer.exl3_dims = (E, H, I)
        layer.exl3_loaded = set()

        def register(name: str, shape: tuple[int, ...], dtype: torch.dtype, dim: int | None, unit: int):
            param = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)

            def loader(p: torch.nn.Parameter, loaded: torch.Tensor, *args, **kwargs) -> None:
                if dim is not None:
                    loaded = loaded.narrow(dim, rank * unit, unit)
                if tuple(loaded.shape) != tuple(p.shape):
                    raise ValueError(f"{name}: checkpoint shape {tuple(loaded.shape)} != {tuple(p.shape)}")
                p.data.copy_(loaded.to(p.dtype))
                layer.exl3_loaded.add(name)

            param.weight_loader = loader  # type: ignore[attr-defined]
            layer.register_parameter(name, param)

        for proj in ("gate", "up"):
            register(f"exl3_{proj}_trellis", (E, H // 16, I // 16, 16 * self.bits), torch.int16, 2, I // 16)
            register(f"exl3_{proj}_suh", (E, H), torch.float16, None, 0)
            register(f"exl3_{proj}_svh", (E, I), torch.float16, 1, I)
        register("exl3_down_trellis", (E, I // 16, H // 16, 16 * self.bits), torch.int16, 1, I // 16)
        register("exl3_down_suh", (E, I), torch.float16, 1, I)
        register("exl3_down_svh", (E, H), torch.float16, None, 0)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from vllm.model_executor.layers.quantization import exl3_sm70_ops as ops

        ops.lib()  # fail at load time, not on the first request
        # With a quantization config set, vLLM does not check for missing parameters: an EXL3 tensor absent from
        # the checkpoint would silently stay uninitialized.
        want = {f"exl3_{p}_{k}" for p in ("gate", "up", "down") for k in ("trellis", "suh", "svh")}
        if layer.exl3_loaded != want:
            raise ValueError(f"EXL3 expert tensors missing from the checkpoint: {sorted(want - layer.exl3_loaded)}")
        layer.exl3_bank = ops.Exl3ExpertBank(
            layer.exl3_gate_trellis.data, layer.exl3_gate_suh.data, layer.exl3_gate_svh.data,
            layer.exl3_up_trellis.data, layer.exl3_up_suh.data, layer.exl3_up_svh.data,
            layer.exl3_down_trellis.data, layer.exl3_down_suh.data, layer.exl3_down_svh.data,
            self.moe.experts_per_token,
        )
        logger.info_once(
            "GLM-5.3 route: SM70 EXL3 (mcg, column-major, K=2/3/4 per layer) routed experts, "
            "native trellis decode + m8n8k4."
        )

    def maybe_make_prepare_finalize(self, routing_tables=None):
        # This method owns permute, expert GEMMs and reduce; do not wrap it in the modular-kernel path.
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2:
            raise TypeError("SM70 EXL3 MoE requires CUDA FP16 activations [M, H].")
        bank = layer.exl3_bank
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return x.new_empty((0, x.shape[1]))
        x = x.contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        w = topk_weights.to(torch.float32).contiguous()
        if num_tokens <= _MAX_TOKENS_PER_LAUNCH:
            return bank.forward(x, ids, w).to(x.dtype)
        out = torch.empty_like(x)
        for s in range(0, num_tokens, _MAX_TOKENS_PER_LAUNCH):
            e = min(s + _MAX_TOKENS_PER_LAUNCH, num_tokens)
            out[s:e] = bank.forward(x[s:e].contiguous(), ids[s:e].contiguous(), w[s:e].contiguous()).to(x.dtype)
        return out

    def apply_monolithic(self, layer, x, router_logits, input_ids=None) -> torch.Tensor:
        raise NotImplementedError("SM70 EXL3 MoE is not monolithic.")

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return None
