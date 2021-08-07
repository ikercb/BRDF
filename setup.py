from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(name="myCode", 
      ext_modules=cythonize('BRDF.pyx', language_level = 3), 
      include_dirs=[numpy.get_include()], zip_safe=False)
