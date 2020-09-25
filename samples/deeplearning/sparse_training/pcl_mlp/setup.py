import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='pcl_mlp',
      py_modules = ['pcl_mlp'],
      ext_modules=[CppExtension('pcl_mlp_ext', ['pcl_mlp_ext.cpp'], extra_compile_args=['-fopenmp', '-g', '-march=native'],
        include_dirs=['{}/include/'.format(os.getenv("LIBXSMM_ROOT"))],
        library_dirs=['{}/lib/'.format(os.getenv("LIBXSMM_ROOT"))],
        libraries=['xsmm'])],
      cmdclass={'build_ext': BuildExtension})

