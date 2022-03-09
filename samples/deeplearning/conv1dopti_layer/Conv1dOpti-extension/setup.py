from setuptools import setup, Extension
from torch.utils import cpp_extension
import os
LIBXSMM_ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
print("LIBXSMM root directory path: ", LIBXSMM_ROOT)

setup(name='conv1dopti-layer',
      ext_modules=[cpp_extension.CppExtension('Conv1dOpti_cpp', ['Conv1dOpti.cpp'], \
                                    author="Narendra Chaudhary", \
                                    author_email="narendra.chaudhary@intel.com", \
                                    description="PyTorch Extension for optimized 1D dilated convolutional layer", \
                                    extra_compile_args=['-O3', '-g', \
                                    '-fopenmp-simd', '-fopenmp', '-march=native',\
                                    # '-mprefer-vector-width=512', '-mavx512f', '-mavx512cd', '-mavx512bw', \
                                    # '-mavx512dq', '-mavx512vl', '-mavx512ifma', '-mavx512vbmi' \
                                    ], \
                                    include_dirs=['{}/include/'.format(LIBXSMM_ROOT)], \
                                    library_dirs=['{}/lib/'.format(LIBXSMM_ROOT)], \
                                    libraries=['xsmm'], \
                                    )],
      py_modules=['Conv1dOpti_ext'],
      cmdclass={'build_ext': cpp_extension.BuildExtension})