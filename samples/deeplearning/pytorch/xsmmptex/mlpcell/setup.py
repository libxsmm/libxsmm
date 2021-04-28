import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='mlpcell',
      py_modules = ['mlpcell'],
      #ext_modules=[CppExtension('mlpcell_ext', ['mlpcell_main.cpp'], extra_compile_args=['-fopenmp', '-g', '-O0', '-march=native'],
      ext_modules=[CppExtension('mlpcell_ext', ['mlpcell_main.cpp'], extra_compile_args=['-fopenmp', '-g', '-O2', '-march=native'],
        include_dirs=['{}/include/'.format(os.getenv("LIBXSMM_ROOT"))],
        library_dirs=['{}/lib/'.format(os.getenv("LIBXSMM_ROOT"))],
        libraries=['xsmm'])],
      cmdclass={'build_ext': BuildExtension})

