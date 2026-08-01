#!/usr/bin/env python
"""spconv-rocm: ROCm port of spconv using FlyDSL GEMM backend."""

import os
import re
from pathlib import Path

from setuptools import setup, find_packages

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")
ROCM_PATH = Path(os.environ.get("ROCM_PATH", "/opt/dtk"))
ROCM_VERSION_FILE = ROCM_PATH / ".info/version-dev"


def _split_env_paths(env_name: str):
    env_value = os.environ.get(env_name, "")
    if not env_value:
        return []
    return [path.strip() for path in env_value.split(os.pathsep) if path.strip()]


def _read_rocm_version_tuple(version_file: Path = ROCM_VERSION_FILE):
    try:
        content = version_file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    match = re.search(r"(\d+)\.(\d+)", content)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _should_define_using_gcn() -> bool:
    version = _read_rocm_version_tuple()
    if version is None:
        return False
    return version < (26, 4)


def _spconv_hip_extension():
    csrc_dir = here / "spconv" / "csrc_hip"
    include_dirs = [str(csrc_dir)]
    include_dirs.extend(_split_env_paths("SPCONV_HYTLASS_INCLUDE"))
    include_dirs.extend(_split_env_paths("SPCONV_EXTRA_INCLUDE_PATHS"))
    cxx_args = ["-O3", "-DROCM_VERSION=60300"]
    nvcc_args = [
        "-O3",
        "-DROCM_VERSION=60300",
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_HALF_OPERATORS__",
    ]
    if _should_define_using_gcn():
        cxx_args.append("-D__USING_GCN__")
        nvcc_args.append("-D__USING_GCN__")
    return CUDAExtension(
        name="spconv._C_hip",
        sources=[
            str(csrc_dir / "indice_pairs_api.cpp"),
            str(csrc_dir / "indice_pairs_kernel.hip"),
        ],
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        },
    )


if _bdist_wheel is not None:
    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False
else:
    bdist_wheel = None

with open("version.txt", "r") as f:
    version = f.read().strip()

version += "+rocm1"

setup(
    name="spconv-rocm",
    version=version,
    description="Spatial sparse convolution — ROCm port (FlyDSL backend)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZJLi2013/spconv_rocm",
    author="Yan Yan, ZJLi2013 (ROCm port)",
    python_requires=">=3.8",
    packages=find_packages(exclude=("tests",)),
    ext_modules=[_spconv_hip_extension()],
    cmdclass={
        "build_ext": BuildExtension,
        **({"bdist_wheel": bdist_wheel} if bdist_wheel is not None else {}),
    },
    # include_package_data=True,
    package_data={
        "spconv": [
            "*.so",
        ],
    },
    install_requires=[
        "torch",
        "numpy",
    ],
    extras_require={
        "implicit-gemm": ["cumm-rocm"],
    },
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
