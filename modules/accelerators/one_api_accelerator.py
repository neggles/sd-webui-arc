import contextlib
import numpy
import torch
import intel_extension_for_pytorch as ipex

from packaging import version
from modules.accelerators.base_accelerator import BaseAccelerator
from modules.sd_hijack_utils import CondFunc


class OneApiAccelerator(BaseAccelerator):
    _instance = None

    def __init__(self):
        self.device = torch.device(self.device_string)
        self.props = ipex.xpu.get_device_properties(device=self.device)
        # ipex.xpu.set_verbose_level(ipex.xpu.VerbLevel.ON)
        return

    @classmethod
    @property
    def device_string(cls):
        return "xpu"

    @classmethod
    def _has_xpu(cls):
        if getattr(torch, "xpu", None) is None:
            return False
        return ipex.xpu.is_available()

    @classmethod
    def discover(cls):
        if cls._has_xpu():
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
        return None

    ## BaseAccelerator methods
    def get_device(self):
        return ipex.xpu.get_device_type()

    def get_available_vram(self):
        return self.get_free_memory()

    def memory_stats(self, device=None):
        return ipex.xpu.memory_stats(device=device if device is not None else self.device)

    def memory_summary(self):
        return ipex.xpu.memory_summary(self.device)

    def get_free_memory(self):
        stats = ipex.xpu.memory_stats(self.device)
        mem_active = stats["active_bytes.all.current"]
        mem_reserved = stats["reserved_bytes.all.current"]
        return self.get_total_memory() - mem_active - mem_reserved

    def get_total_memory(self):
        return self.props.total_memory

    def reset_peak_memory_stats(self):
        return ipex.xpu.reset_peak_memory_stats(self.device)

    def empty_cache(self):
        ipex.xpu.empty_cache()

    def gc(self):
        self.empty_cache()

    def enable_tf32(self):
        if self.props.dev_type == "gpu":
            ipex.xpu.set_fp32_math_mode(mode=ipex.FP32MathMode.TF32)
            return
        else:
            print(f"device {self.device} is not a gpu, not enabling tf32 mode")
            return

    def get_rng_state_all(self):
        return ipex.xpu.get_rng_state_all()

    def set_rng_state(self, state):
        ipex.xpu.set_rng_state_all(state)

    def manual_seed(self, seed):
        torch.manual_seed(seed)
        ipex.xpu.manual_seed_all(seed)
        numpy.random.seed(seed)

    @classmethod
    @property
    def amp(self):
        return ipex.xpu.amp

    ## Expected by the accelerators module
    def autocast(self, *args, **kwargs):
        return torch.autocast("xpu", *args, **kwargs)

    def without_autocast(self, disable: bool):
        return (
            torch.autocast("xpu", enabled=False)
            if ipex.xpu.is_autocast_xpu_enabled() and not disable
            else contextlib.nullcontext()
        )

    def optimize(self, model, dtype: torch.dtype):
        return ipex.optimize(model, dtype)

    def randn(self, shape, device=torch.device("cpu")):
        return torch.randn(shape, device=device).to(self.device)

    # Remaining methods are called by exported methods above or other modules
    def randn_like(self, x, device=None):
        if device is None:
            device = self.device
        return torch.randn_like(x, device=torch.device("cpu")).to(x.device)

    def get_einsum_op_mem(self):
        return self.get_free_memory() / 3 / (1 << 20)


# xpu workaround for https://github.com/pytorch/pytorch/issues/89784
def cumsum_fix(input, cumsum_func, *args, **kwargs):
    if input.device.type == "xpu":
        output_dtype = kwargs.get("dtype", input.dtype)
        if output_dtype == torch.int64:
            return cumsum_func(input.cpu(), *args, **kwargs).to(input.device)
        elif (
            cumsum_needs_bool_fix
            and output_dtype == torch.bool
            or cumsum_needs_int_fix
            and (output_dtype == torch.int8 or output_dtype == torch.int16)
        ):
            return cumsum_func(input.to(torch.int32), *args, **kwargs).to(torch.int64)
    return cumsum_func(input, *args, **kwargs)


if ipex.xpu.is_available():
    # fix for randn in torchsde
    CondFunc(
        "torchsde._brownian.brownian_interval._randn",
        lambda _, size, dtype, device, seed: torch.randn(
            size,
            dtype=dtype,
            device=torch.device("cpu"),
            generator=torch.Generator(torch.device("cpu")).manual_seed(int(seed)),
        ).to(device),
        lambda _, size, dtype, device, seed: device.type == "xpu",
    )

    if version.parse(torch.__version__) > version.parse("1.13.1"):
        cumsum_needs_int_fix = (
            not torch.Tensor([1, 2])
            .to(torch.device("xpu"))
            .equal(torch.ShortTensor([1, 1]).to(torch.device("xpu")).cumsum(0))
        )
        cumsum_needs_bool_fix = (
            not torch.BoolTensor([True, True])
            .to(device=torch.device("xpu"), dtype=torch.int64)
            .equal(torch.BoolTensor([True, False]).to(torch.device("xpu")).cumsum(0))
        )
        cumsum_fix_func = lambda orig_func, input, *args, **kwargs: cumsum_fix(
            input, orig_func, *args, **kwargs
        )
        CondFunc("torch.cumsum", cumsum_fix_func, None)
        CondFunc("torch.Tensor.cumsum", cumsum_fix_func, None)
        CondFunc("torch.narrow", lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).clone(), None)
