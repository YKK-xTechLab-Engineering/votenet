from setuptools import setup, find_packages
from pathlib import Path
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = Path(__file__).parent
EXT_ROOT = ROOT_DIR / "votenet" / "pointnet2" / "_ext_src"
SRC_DIR = EXT_ROOT / "src"
INCLUDE_DIR = EXT_ROOT / "include"

sources_abs = list(SRC_DIR.glob("*.cpp")) + list(SRC_DIR.glob("*.cu"))
# setuptools requires sources to be relative to setup.py dir
sources = [os.path.relpath(str(p), start=str(ROOT_DIR)) for p in sources_abs]

setup(
    name="votenet",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name="votenet.pointnet2._ext",
            sources=sources,
            include_dirs=[str(INCLUDE_DIR)],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)