#!/usr/bin/env python3
# coding: utf-8
from distutils.core import setup, Extension
import numpy as np

ext = Extension('roc_auc_score', sources=["rac.cpp"], extra_compile_args=['-fopenmp', '-g', '-march=native', '-std=c++14', '-O2'])
setup(name="roc_auc_score", include_dirs=[np.get_include], ext_modules=[ext])


