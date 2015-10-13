#!/bin/bash

if [[ "-cp2k" == "$1" ]] ; then
  shift
  make -e $* ALIGNED_STORES=1 MNK=" \
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
    13 26 28 32 45, \
    7 13 25 32"
elif [[ "-cia" == "$1" ]] ; then
  shift
  make -e $* MNK=" \
    0 8 15, \
    23 24 42"
elif [[ "-cib" == "$1" ]] ; then
  shift
  make -e $* ROW_MAJOR=1 MNK=" \
    0 8 15, \
    23 24 42"
else
  make -e $*
fi

