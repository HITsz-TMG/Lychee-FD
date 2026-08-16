from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Literal

import torch
import torch.nn as nn

_SCALED_NVFP4_QUANT = None
_CUTLASS_SCALED_NVFP4_MM = None
_LIGHTX2V_KERNEL_IMPORT_ERROR: Optional[BaseException] = None


def _load_lightx2v_kernel():
    global _SCALED_NVFP4_QUANT, _CUTLASS_SCALED_NVFP4_MM
    global _LIGHTX2V_KERNEL_IMPORT_ERROR

    if _SCALED_NVFP4_QUANT is not None and _CUTLASS_SCALED_NVFP4_MM is not None:
        return _SCALED_NVFP4_QUANT, _CUTLASS_SCALED_NVFP4_MM

    try:
        from lightx2v_kernel.gemm import scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm
    except ImportError as exc:
        _LIGHTX2V_KERNEL_IMPORT_ERROR = exc
        raise RuntimeError("lightx2v_kernel not found. Cannot use FP4Linear.") from exc

    _SCALED_NVFP4_QUANT = scaled_nvfp4_quant
    _CUTLASS_SCALED_NVFP4_MM = cutlass_scaled_nvfp4_mm
    _LIGHTX2V_KERNEL_IMPORT_ERROR = None
    return _SCALED_NVFP4_QUANT, _CUTLASS_SCALED_NVFP4_MM


@dataclass(frozen=True)
class FP4GemmOptions:
    cast_inputs: bool = True
    cast_output_back: bool = True
    fp16_weight_storage: Literal["keep", "cpu_offload", "discard"] = "discard"
    materialize_fp4_on_wrap: bool = True


class FP4Linear(nn.Module):
    def __init__(self, linear: nn.Linear, *, options: FP4GemmOptions):
        super().__init__()
        _load_lightx2v_kernel()
        
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear)}")

        if options.fp16_weight_storage not in ("keep", "cpu_offload", "discard"):
            raise ValueError(
                f"invalid fp16_weight_storage={options.fp16_weight_storage!r}; "
                "expected one of {'keep','cpu_offload','discard'}"
            )

        self.linear: Optional[nn.Linear] = linear if options.fp16_weight_storage == "keep" else None
        self.options = options

        self._fp16_weight_cpu: Optional[torch.Tensor] = None
        self._fp16_bias_cpu: Optional[torch.Tensor] = None

        self.bias: Optional[nn.Parameter] = None
        if options.fp16_weight_storage != "keep":
            self.bias = (nn.Parameter(linear.bias.detach().clone())
                         if linear.bias is not None else None)
            self._fp16_weight_cpu = linear.weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            if linear.bias is not None:
                self._fp16_bias_cpu = linear.bias.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()

        self.register_buffer("_fp4_weight", None, persistent=False)
        self.register_buffer("_fp4_weight_scale", None, persistent=False)
        self.register_buffer("_w_global_scale", None, persistent=False)
        self._weight_cache_device: Optional[torch.device] = None
        self._last_weight_version: Optional[int] = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, options: FP4GemmOptions) -> "FP4Linear":
        return cls(linear, options=options)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        if self.linear is not None:
            src_weight = self.linear.weight.detach()
            src_bias = self.linear.bias.detach() if self.linear.bias is not None else None
        elif self._fp16_weight_cpu is not None:
            src_weight = self._fp16_weight_cpu.detach()
            src_bias = self._fp16_bias_cpu.detach() if self._fp16_bias_cpu is not None else None
        else:
            raise RuntimeError("FP4Linear cannot be deep-copied without an FP16 weight source.")

        linear = nn.Linear(
            in_features=src_weight.shape[1],
            out_features=src_weight.shape[0],
            bias=src_bias is not None,
            device=src_weight.device,
            dtype=src_weight.dtype,
        )
        linear.weight.data.copy_(src_weight)
        if src_bias is not None:
            linear.bias.data.copy_(src_bias)

        cloned = FP4Linear(linear, options=self.options)
        memo[id(self)] = cloned

        if self._fp16_weight_cpu is not None:
            cloned._fp16_weight_cpu = self._fp16_weight_cpu.detach().clone()
        if self._fp16_bias_cpu is not None:
            cloned._fp16_bias_cpu = self._fp16_bias_cpu.detach().clone()

        if self._fp4_weight is not None:
            cloned._fp4_weight = self._fp4_weight.detach().clone()
        if self._fp4_weight_scale is not None:
            cloned._fp4_weight_scale = self._fp4_weight_scale.detach().clone()
        if self._w_global_scale is not None:
            cloned._w_global_scale = self._w_global_scale.detach().clone()

        cloned._weight_cache_device = self._weight_cache_device
        cloned._last_weight_version = self._last_weight_version
        return cloned

    def invalidate_weight_cache(self) -> None:
        self._fp4_weight = None
        self._fp4_weight_scale = None
        self._w_global_scale = None
        self._weight_cache_device = None
        self._last_weight_version = None

    def _cached_fp4_device(self) -> Optional[torch.device]:
        if self._fp4_weight is None or self._fp4_weight_scale is None:
            return None
        if self._fp4_weight.device != self._fp4_weight_scale.device:
            return None
        return self._fp4_weight.device

    def materialize_fp4_weight(self, device: torch.device) -> None:
        self._maybe_requantize_weight(device)

    def _maybe_requantize_weight(self, device: torch.device) -> None:
        device = torch.device(device)
        cache_device = self._cached_fp4_device()
        version: Optional[int] = None
        if self.linear is not None:
            weight = self.linear.weight
            v = getattr(weight, "_version", None)
            version = v if isinstance(v, int) else None
            if (self._fp4_weight is not None and self._fp4_weight_scale is not None
                    and cache_device == device
                    and (version is None or version == self._last_weight_version)):
                return
        else:
            if (self._fp4_weight is not None and self._fp4_weight_scale is not None
                    and cache_device == device):
                return

        if self.linear is not None:
            w_src = self.linear.weight.detach()
        elif self._fp16_weight_cpu is not None:
            w_src = self._fp16_weight_cpu
        else:
            raise RuntimeError(
                "FP4Linear has no FP16 weight source available to (re)quantize."
            )

        w_n_k = w_src.to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()

        w_max = w_n_k.abs().max().clamp(min=1e-5)
        w_global_scale = (2688.0 / w_max).to(torch.float32)
        scaled_nvfp4_quant, _ = _load_lightx2v_kernel()
        with torch.cuda.device(device):
            qweight, w_scale = scaled_nvfp4_quant(w_n_k, w_global_scale)

        self._fp4_weight = qweight
        self._fp4_weight_scale = w_scale
        self._w_global_scale = w_global_scale
        self._weight_cache_device = device
        self._last_weight_version = version

        if self.options.fp16_weight_storage == "discard":
            self._fp16_weight_cpu = None
            self._fp16_bias_cpu = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            if self.linear is not None:
                return self.linear(x)
            if self._fp16_weight_cpu is not None:
                bias = self._fp16_bias_cpu
                return torch.nn.functional.linear(x, self._fp16_weight_cpu.to(dtype=x.dtype),
                                                  bias.to(dtype=x.dtype) if bias is not None else None)
            raise RuntimeError("FP4Linear cannot run on CPU because FP16 weights are not kept.")

        in_dtype = x.dtype
        if in_dtype not in (torch.float16, torch.bfloat16):
            if not self.options.cast_inputs:
                if self.linear is not None:
                    return self.linear(x)
                if self._fp16_weight_cpu is not None:
                    w = self._fp16_weight_cpu.to(device=x.device, dtype=in_dtype)
                    b = self._fp16_bias_cpu
                    b = b.to(device=x.device, dtype=in_dtype) if b is not None else None
                    return torch.nn.functional.linear(x, w, b)
                raise RuntimeError("cast_inputs=False requires FP16 weights for fallback, but they were discarded.")
            x_fp = x.to(torch.bfloat16)
            out_dtype = torch.bfloat16
        else:
            x_fp = x
            out_dtype = in_dtype

        self._maybe_requantize_weight(x_fp.device)

        if self.linear is not None:
            bias = self.linear.bias
        else:
            bias = self.bias
        if bias is not None:
            if bias.device != x_fp.device:
                bias = bias.to(device=x_fp.device, non_blocking=True)
            if bias.dtype != out_dtype:
                bias = bias.to(dtype=out_dtype)

        m = x_fp.shape[:-1]
        k = x_fp.shape[-1]
        x_2d = x_fp.view(-1, k)
        
        x_max = x_2d.abs().max().clamp(min=1e-5)
        input_global_scale = (2688.0 / x_max).to(torch.float32)
        scaled_nvfp4_quant, cutlass_scaled_nvfp4_mm = _load_lightx2v_kernel()
        with torch.cuda.device(x_fp.device):
            q_x, s_x = scaled_nvfp4_quant(x_2d, input_global_scale)

            alpha = 1.0 / (input_global_scale * self._w_global_scale)
            y_2d = cutlass_scaled_nvfp4_mm(
                q_x,
                self._fp4_weight,
                s_x,
                self._fp4_weight_scale,
                alpha=alpha,
                bias=bias,
            )
        
        y = y_2d.view(*m, -1)

        if self.options.cast_inputs and self.options.cast_output_back and y.dtype != in_dtype:
            return y.to(in_dtype)
        return y


def enable_fp4_gemm(
    model: nn.Module,
    *,
    options: FP4GemmOptions = FP4GemmOptions(),
    module_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    inplace: bool = True,
) -> nn.Module:
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    def should_wrap(name: str, m: nn.Module) -> bool:
        if not isinstance(m, nn.Linear):
            return False
        if module_filter is None:
            return True
        return bool(module_filter(name, m))

    def _recurse(prefix: str, parent: nn.Module) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if should_wrap(full_name, child):
                fp4_mod = FP4Linear.from_linear(child, options=options)
                if options.materialize_fp4_on_wrap and child.weight.is_cuda:
                    fp4_mod.materialize_fp4_weight(child.weight.device)
                setattr(parent, child_name, fp4_mod)
            else:
                _recurse(full_name, child)

    _recurse("", model)
    return model
