"""In-memory stand-ins for tensorrt and pycuda, so the board-only paths run under pytest.

The fake engine has the exported yolov8 I/O (images 1x3x640x640 -> output0
1x84x8400) and fills its output with the input mean, so a test can tell the
whole copy chain worked. The fake builder drives a calibrator the way TensorRT
does: read the cache, else consume batches until None, then write it.
"""
import sys
import types

import numpy as np


class Mem:
    registry = {}

    def __init__(self, nbytes):
        self.buf = np.zeros(nbytes, np.uint8)
        self.freed = False
        Mem.registry[id(self)] = self

    def __int__(self):
        return id(self)

    def free(self):
        self.freed = True


class Stream:
    handle = 1

    def synchronize(self):
        pass


class Event:
    def record(self, stream):
        pass

    def time_since(self, other):
        return 1.5


def memcpy_htod(dev, host):
    raw = np.ascontiguousarray(host).view(np.uint8).ravel()
    dev.buf[: raw.size] = raw


def memcpy_dtoh(host, dev):
    host[...] = dev.buf[: host.nbytes].view(host.dtype).reshape(host.shape)


driver = types.ModuleType("pycuda.driver")
driver.mem_alloc = Mem
driver.pagelocked_empty = lambda n, dtype: np.empty(n, dtype)
driver.memcpy_htod = memcpy_htod
driver.memcpy_dtoh = memcpy_dtoh
driver.memcpy_htod_async = lambda dev, host, stream: memcpy_htod(dev, host)
driver.memcpy_dtoh_async = lambda host, dev, stream: memcpy_dtoh(host, dev)
driver.Stream = Stream
driver.Event = Event
pycuda = types.ModuleType("pycuda")
pycuda.driver = driver
pycuda.autoinit = types.ModuleType("pycuda.autoinit")

IO = {"images": ((1, 3, 640, 640), np.float32, "input"), "output0": ((1, 84, 8400), np.float32, "output")}


class Context:
    def __init__(self):
        self.addr = {}

    def set_tensor_address(self, name, ptr):
        self.addr[name] = ptr

    def execute_async_v3(self, stream_handle):
        x = Mem.registry[self.addr["images"]].buf[: 3 * 640 * 640 * 4].view(np.float32)
        y = np.full(84 * 8400, x.mean(), np.float32)
        memcpy_htod(Mem.registry[self.addr["output0"]], y)
        return True


class Engine:
    num_io_tensors = 2

    def get_tensor_name(self, i):
        return list(IO)[i]

    def get_tensor_shape(self, n):
        return IO[n][0]

    def get_tensor_dtype(self, n):
        return IO[n][1]

    def get_tensor_mode(self, n):
        return IO[n][2]

    def create_execution_context(self):
        return Context()


class Runtime:
    def __init__(self, logger):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def deserialize_cuda_engine(self, data):
        return Engine() if data == b"fake-engine" else None


class Tensor:
    def __init__(self, dtype):
        self.dtype = dtype


class Layer:
    def __init__(self, name, kind="conv", dtype=np.float32):
        self.name, self.type, self.num_outputs = name, kind, 1
        self.out = Tensor(dtype)
        self.precision = None

    def get_output(self, j):
        return self.out


LAYERS = [
    ("/model.0/conv/Conv", "conv", np.float32),
    ("/model.0/act/Sigmoid", "act", np.float32),
    ("/model.22/cv2.0/cv2.0.0/conv/Conv", "conv", np.float32),
    ("/model.22/cv3.0/cv3.0.0/conv/Conv", "conv", np.float32),
    ("/model.22/Constant_1", "constant", np.int32),
    ("/model.22/Shape", "shape", np.int32),
    ("/model.22/Concat_3", "concat", np.float32),
    ("(Unnamed Layer* 3) [Shuffle]", "shuffle", np.float32),
]


class Network:
    def __init__(self):
        self.layers = [Layer(*spec) for spec in LAYERS]

    @property
    def num_layers(self):
        return len(self.layers)

    def get_layer(self, i):
        return self.layers[i]


class OnnxParser:
    num_errors = 0

    def __init__(self, network, logger):
        pass

    def parse(self, data):
        return data != b"bad"


class Config:
    def __init__(self):
        self.flags = set()
        self.int8_calibrator = None

    def set_flag(self, f):
        self.flags.add(f)

    def set_memory_pool_limit(self, pool, n):
        self.workspace = n


class Builder:
    built = []

    def __init__(self, logger):
        pass

    def create_network(self, flags):
        return Network()

    def create_builder_config(self):
        return Config()

    def build_serialized_network(self, network, config):
        cal = config.int8_calibrator
        batches = 0
        if cal is not None and cal.read_calibration_cache() is None:
            while cal.get_batch(["images"]) is not None:
                batches += 1
            cal.write_calibration_cache(b"fake-cache")
        Builder.built.append({"flags": set(config.flags), "pinned": [l.name for l in network.layers if l.precision == "half"], "batches": batches})
        return b"fake-engine"


tensorrt = types.ModuleType("tensorrt")
tensorrt.__version__ = "10.3.0-fake"
tensorrt.Logger = type("Logger", (), {"INFO": "info", "WARNING": "warning", "ERROR": "error", "__init__": lambda self, level=None: None})
tensorrt.DataType = type("DataType", (), {"HALF": "half"})
tensorrt.TensorIOMode = type("TensorIOMode", (), {"INPUT": "input", "OUTPUT": "output"})
tensorrt.LayerType = type("LayerType", (), {"CONSTANT": "constant", "SHAPE": "shape"})
tensorrt.BuilderFlag = type("BuilderFlag", (), {"FP16": "fp16", "INT8": "int8", "OBEY_PRECISION_CONSTRAINTS": "obey"})
tensorrt.MemoryPoolType = type("MemoryPoolType", (), {"WORKSPACE": "workspace"})
tensorrt.IInt8EntropyCalibrator2 = type("IInt8EntropyCalibrator2", (), {"__init__": lambda self: None})
tensorrt.float32 = np.float32
tensorrt.nptype = lambda dt: dt
tensorrt.Runtime = Runtime
tensorrt.Builder = Builder
tensorrt.OnnxParser = OnnxParser


def install(monkeypatch):
    Mem.registry.clear()
    Builder.built.clear()
    for name, mod in (("tensorrt", tensorrt), ("pycuda", pycuda), ("pycuda.driver", driver), ("pycuda.autoinit", pycuda.autoinit)):
        monkeypatch.setitem(sys.modules, name, mod)
