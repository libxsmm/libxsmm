import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='pcl_embedding_bag',
      py_modules = ['pcl_embedding_bag'],
      #ext_modules=[CppExtension('pcl_embedding_bag_cpp', ['pcl_embedding_bag.cpp'], extra_compile_args=['-fopenmp', '-g', '-march=native'])],
      ext_modules=[CppExtension('pcl_embedding_bag_cpp', ['pcl_embedding_bag.cpp'],
        #extra_compile_args=['-fopenmp', '-g', '-mavx512f', '-mrtm', '-mf16c', '-mavx512bw', '-mavx512vl'],
        extra_compile_args=['-fopenmp', '-g', '-mavx2', '-mrtm'],
        include_dirs=['{}/include/'.format(os.getenv("LIBXSMM_ROOT"))],
        library_dirs=['{}/lib/'.format(os.getenv("LIBXSMM_ROOT"))],
        libraries=['xsmm'])],

      cmdclass={'build_ext': BuildExtension})

