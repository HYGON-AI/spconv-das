#!/usr/bin/env python
"""spconv-rocm: ROCm port of spconv using FlyDSL GEMM backend."""

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")

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
    include_package_data=True,
    package_data={
        "spconv": [
            "csrc_hip/*.cpp",
            "csrc_hip/*.hip",
            "csrc_hip/*.h",
            "csrc_hip/*.hpp",
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
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
