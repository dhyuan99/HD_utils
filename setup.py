# python setup.py build_ext --inplace

from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("quantize.pyx"),
    include_dirs=[numpy.get_include()]
)  