#!/bin/sh

make clean; make AVX=3 OMP=1 OPT=3 MIC=0
