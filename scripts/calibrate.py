import json
from pathlib import Path

import numpy as np

from bench.preprocess import preprocess


def load_manifest(path):
    return json.loads(Path(path).read_text())


def entropy_calibrator(manifest, images_dir, cache, batch_size, imgsz):
    """IInt8EntropyCalibrator2 over the manifest's images; writes/reads the trtexec-compatible cache."""
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    files = [Path(images_dir) / f for f in load_manifest(manifest)["files"]]
    if not files:
        raise RuntimeError(f"empty manifest {manifest}")
    cache = Path(cache)

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.i = 0
            self.n_batches = len(files) // batch_size
            self.host = np.empty((batch_size, 3, imgsz, imgsz), dtype=np.float32)
            self.dev = cuda.mem_alloc(self.host.nbytes)

        def get_batch_size(self):
            return batch_size

        def get_batch(self, names):
            if self.i >= self.n_batches:
                return None
            for j, path in enumerate(files[self.i * batch_size : (self.i + 1) * batch_size]):
                self.host[j] = preprocess(path, imgsz)[0][0]
            cuda.memcpy_htod(self.dev, self.host)
            self.i += 1
            if self.i % 10 == 0 or self.i == self.n_batches:
                print(f"  calib batch {self.i}/{self.n_batches}")
            return [int(self.dev)]

        def read_calibration_cache(self):
            # an existing cache skips the image pass entirely; delete it to recalibrate
            if cache.exists():
                print(f"  reusing {cache}")
                return cache.read_bytes()
            return None

        def write_calibration_cache(self, data):
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
            print(f"  wrote {cache} ({len(data)} bytes)")

        def free(self):
            # the builder keeps no reference after serialization; without this a 25-engine
            # sweep leaks a batch buffer per build, ~1 GB on an 8 GB board
            self.dev.free()

    return Calibrator()
