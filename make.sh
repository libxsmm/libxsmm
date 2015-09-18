#!/bin/bash

if [[ "-cp2k" == "$1" ]] ; then
  shift
  make $* ROW_MAJOR=0 ALIGNED_STORES=1 PREFETCH=1 MNK=" \
    23, \
    6, \
    14 16 29, \
    14 32 29, \
    5 32 13 24 26, \
    9 32 22, \
    32, \
    64, \
    78, \
    16 29 55, \
    32 29 55, \
    12, \
    13 26 28 32 45"
else
  make $*
fi

