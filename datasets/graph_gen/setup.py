from torch.utils.cpp_extension import CppExtension, BuildExtension
from setuptools import setup

ext_modules = [
    CppExtension(
        "matrix_neighbour",
        ["matrix_neighbour.cpp"],
        extra_compile_args=["-std=c++17", "-O3", "-march=native"],
    ),
]

setup(
    name="matrix_neighbour",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
