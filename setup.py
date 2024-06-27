from setuptools import setup, Extension
import pybind11
import os
import sys

extra_compile_args = ["-std=c++11"]
extra_link_args = []

if sys.platform == "darwin":  # macOS
    llvm_include = os.popen("brew --prefix llvm").read().strip() + "/include"
    llvm_lib = os.popen("brew --prefix llvm").read().strip() + "/lib"
    extra_compile_args += ["-Xpreprocessor", "-fopenmp", "-I" + llvm_include]
    extra_link_args += ["-lomp", "-L" + llvm_lib]
elif sys.platform == "linux" or sys.platform == "linux2":  # Linux
    extra_compile_args += ["-fopenmp"]
    extra_link_args += ["-fopenmp"]

ext_modules = [
    Extension(
        "compute_scores",
        ["compute_scores.cpp", "bindings.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
]

setup(
    name="compute_scores",
    ext_modules=ext_modules,
    zip_safe=False,
)
