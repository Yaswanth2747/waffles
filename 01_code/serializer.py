# serializer.py

import struct
import numpy as np
import torch


def serialize_tensor(tensor):

    arr = tensor.detach().cpu().numpy().astype(np.float32)

    shape = arr.shape
    ndim = len(shape)

    header = struct.pack("I", ndim) + struct.pack(f"{ndim}I", *shape)

    return header + arr.tobytes()


def deserialize_tensor(data):

    if len(data) < 4:
        raise RuntimeError("Payload too small to decode tensor ndim")

    ndim = struct.unpack("I", data[:4])[0]
    shape_size = 4 * ndim
    shape_end = 4 + shape_size

    if len(data) < shape_end:
        raise RuntimeError("Payload too small to decode tensor shape")

    shape = struct.unpack(f"{ndim}I", data[4:shape_end])

    payload = data[shape_end:]

    expected = int(np.prod(shape)) * 4

    if len(payload) != expected:
        raise RuntimeError(
            f"Tensor size mismatch: got {len(payload)} expected {expected}"
        )

    arr = np.frombuffer(payload, dtype=np.float32)

    arr = arr.reshape(shape)

    return torch.tensor(arr)
