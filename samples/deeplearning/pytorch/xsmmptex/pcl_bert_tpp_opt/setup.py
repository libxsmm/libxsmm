import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

if "LIBXSMM_ROOT" not in os.environ:
    raise Exception("LIBXSMM_ROOT is not set! Please point it to libxsmm base directory.")

setup(name='pcl_bert',
      py_modules = ['pcl_bert', 'blocked_layout'],
      version='0.1',
      #ext_modules=[CppExtension('pcl_bert_ext', ['pcl_bert_ext.cpp'], extra_compile_args=['-fopenmp', '-g', '-march=native'],
      #ext_modules=[CppExtension('pcl_bert_ext', ['pcl_bert_ext.cpp', 'pcl_bert_bf16_ext.cpp'], extra_compile_args=['-fopenmp', '-g', '-march=native', '-mavx512f', '-mavx512bw', '-mavx512vl'],
      #ext_modules=[CppExtension('pcl_bert_ext', ['pcl_bert_uni_ext.cpp'], extra_compile_args=['-fopenmp', '-g', '-march=native', '-mavx512f', '-mavx512bw', '-mavx512vl'],
      ext_modules=[CppExtension('pcl_bert_ext', ['pcl_bert_uni_ext.cpp'], extra_compile_args=['-fopenmp', '-g', '-mavx2'],
        include_dirs=['{}/include/'.format(os.getenv("LIBXSMM_ROOT"))],
        library_dirs=['{}/lib/'.format(os.getenv("LIBXSMM_ROOT"))],
        libraries=['xsmm'])],
      cmdclass={'build_ext': BuildExtension})

