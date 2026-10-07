from pathlib import Path

import numpy as np


class OnnxRuntime:
    name = "onnxruntime"

    def __init__(self, path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = tuple(inp.shape)
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, x):
        return self.session.run([self.output_name], {self.input_name: x.astype(np.float32, copy=False)})[0]

    def close(self):
        pass


class TensorRT:
    """TensorRT 10 (JetPack 6) engine runner. infer() spans H2D, execute, D2H and a sync."""

    name = "tensorrt"

    def __init__(self, path):
        import pycuda.autoinit  # noqa: F401  creates the CUDA context
        import pycuda.driver as cuda
        import tensorrt as trt

        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"cannot deserialize {path}: built on another TensorRT or GPU?")
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.io = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise RuntimeError(f"dynamic shape on {name}: {shape}; export with batch=1 and dynamic=False")
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.ctx.set_tensor_address(name, int(dev))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.io[name] = (host, dev, shape, dtype, is_input)
        inputs = [n for n, t in self.io.items() if t[4]]
        outputs = [n for n, t in self.io.items() if not t[4]]
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"expected one input and one output, got {inputs} / {outputs}")
        self.input_name, self.output_name = inputs[0], outputs[0]
        self.input_shape = self.io[self.input_name][2]

    def infer(self, x):
        host_in, dev_in, _, dtype, _ = self.io[self.input_name]
        host_out, dev_out, out_shape, _, _ = self.io[self.output_name]
        np.copyto(host_in, np.ascontiguousarray(x, dtype=dtype).ravel())
        self.cuda.memcpy_htod_async(dev_in, host_in, self.stream)
        if not self.ctx.execute_async_v3(stream_handle=self.stream.handle):
            raise RuntimeError("execute_async_v3 failed")
        self.cuda.memcpy_dtoh_async(host_out, dev_out, self.stream)
        self.stream.synchronize()
        return host_out.reshape(out_shape).astype(np.float32, copy=False)

    def close(self):
        for _, dev, *_ in self.io.values():
            dev.free()
        self.io.clear()


def load(path):
    suffix = Path(path).suffix
    if suffix == ".onnx":
        return OnnxRuntime(path)
    if suffix == ".engine":
        return TensorRT(path)
    raise ValueError(f"{path}: expected .onnx or .engine")
