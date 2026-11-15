"""Read a GPU's energy counter, power draw and SM clock through NVML with ctypes (no pynvml needed).

`Meter(uuid)` opens the device whose UUID torch reports (NVML ignores CUDA_VISIBLE_DEVICES, so the index cannot be
used). `meter.energy_mj()` is nvmlDeviceGetTotalEnergyConsumption (millijoules since the driver loaded);
`meter.sample()` starts a thread that records power (W) and SM clock (MHz) every `period_s` until `stop()` returns
the samples. Energy over a window is the counter's difference; the sampled power is for mean/peak reporting only.
Raises RuntimeError when NVML is missing or a call fails (for example, a counter the user may not read).
"""
import ctypes
import threading
import time

NVML_CLOCK_SM = 1


class Meter:
    """One NVML device, found by UUID."""

    def __init__(self, uuid: str):
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError as e:
            raise RuntimeError(f"NVML not available: {e}") from e
        self._call("nvmlInit_v2")
        self.handle = ctypes.c_void_p()
        self._call("nvmlDeviceGetHandleByUUID", ctypes.c_char_p(uuid.encode()), ctypes.byref(self.handle))
        self._samples = None
        self._thread = None

    def _call(self, name, *args):
        rc = getattr(self.lib, name)(*args)
        if rc != 0:
            raise RuntimeError(f"{name} failed with NVML error {rc}")

    def energy_mj(self) -> int:
        v = ctypes.c_ulonglong()
        self._call("nvmlDeviceGetTotalEnergyConsumption", self.handle, ctypes.byref(v))
        return v.value

    def power_w(self) -> float:
        v = ctypes.c_uint()
        self._call("nvmlDeviceGetPowerUsage", self.handle, ctypes.byref(v))
        return v.value / 1000.0

    def sm_mhz(self) -> int:
        v = ctypes.c_uint()
        self._call("nvmlDeviceGetClockInfo", self.handle, NVML_CLOCK_SM, ctypes.byref(v))
        return v.value

    def sample(self, period_s: float = 0.02) -> None:
        """Start recording (time, watts, SM MHz) in a background thread."""
        self._samples = []
        self._stop = threading.Event()

        def run():
            while not self._stop.is_set():
                self._samples.append((time.perf_counter(), self.power_w(), self.sm_mhz()))
                time.sleep(period_s)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop recording and return the samples."""
        self._stop.set()
        self._thread.join()
        return self._samples


def torch_uuid(device: int = 0) -> str:
    """NVML-style UUID ('GPU-...') of a torch device."""
    import torch
    u = str(torch.cuda.get_device_properties(device).uuid)
    return u if u.startswith("GPU-") else "GPU-" + u
