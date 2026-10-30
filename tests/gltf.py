"""Minimal GLB reader for asserting on exported glTF content in tests."""

import io
import json
import struct

import numpy as np
from PIL import Image

_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def parse(glb: bytes) -> tuple[dict, bytes]:
    """Split a GLB into its JSON chunk and binary chunk."""
    n = struct.unpack("<I", glb[12:16])[0]
    return json.loads(glb[20 : 20 + n]), glb[20 + n + 8 :]


def _view(gltf: dict, bin_: bytes, index: int) -> bytes:
    view = gltf["bufferViews"][index]
    start = view.get("byteOffset", 0)
    return bin_[start : start + view["byteLength"]]


_DTYPES = {5126: np.float32, 5121: np.uint8}


def accessor(gltf: dict, bin_: bytes, index: int) -> np.ndarray:
    """Float (or uint8) accessor data as [count, components]."""
    acc = gltf["accessors"][index]
    data = _view(gltf, bin_, acc["bufferView"])[acc.get("byteOffset", 0) :]
    size = _COMPONENTS[acc["type"]]
    dtype = _DTYPES[acc["componentType"]]
    return np.frombuffer(data, dtype, acc["count"] * size).reshape(-1, size)


def primitives(glb: bytes, mode: int) -> list[dict]:
    """Primitives of one glTF mode (0 points, 1 lines, 4 triangles), each with its
    POSITION data under "positions"."""
    gltf, bin_ = parse(glb)
    return [
        {**prim, "positions": accessor(gltf, bin_, prim["attributes"]["POSITION"])}
        for mesh in gltf["meshes"]
        for prim in mesh["primitives"]
        if prim.get("mode", 4) == mode
    ]


def line_colors(glb: bytes) -> list[tuple[int, int, int]]:
    """The RGB vertex colour (COLOR_0) of each line primitive's first vertex."""
    gltf, bin_ = parse(glb)
    return [
        tuple(int(c) for c in accessor(gltf, bin_, p["attributes"]["COLOR_0"])[0][:3])
        for mesh in gltf["meshes"]
        for p in mesh["primitives"]
        if p.get("mode", 4) == 1 and "COLOR_0" in p["attributes"]
    ]


def textured_quad(glb: bytes) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """The single triangle primitive: positions, raw glTF UVs, material, and its
    emissive texture image."""
    gltf, bin_ = parse(glb)
    (prim,) = [
        p for m in gltf["meshes"] for p in m["primitives"] if p.get("mode", 4) == 4
    ]
    material = gltf["materials"][prim["material"]]
    texture = gltf["textures"][material["emissiveTexture"]["index"]]
    png = _view(gltf, bin_, gltf["images"][texture["source"]]["bufferView"])
    return (
        accessor(gltf, bin_, prim["attributes"]["POSITION"]),
        accessor(gltf, bin_, prim["attributes"]["TEXCOORD_0"]),
        material,
        np.asarray(Image.open(io.BytesIO(png)).convert("RGB")),
    )


def flat_triangles(glb: bytes) -> list[tuple[np.ndarray, dict]]:
    """Untextured triangle primitives: (positions, material) each."""
    gltf, bin_ = parse(glb)
    return [
        (
            accessor(gltf, bin_, prim["attributes"]["POSITION"]),
            gltf["materials"][prim["material"]],
        )
        for mesh in gltf["meshes"]
        for prim in mesh["primitives"]
        if prim.get("mode", 4) == 4 and "TEXCOORD_0" not in prim["attributes"]
    ]
