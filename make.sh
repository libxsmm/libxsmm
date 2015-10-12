#!/bin/bash

if [[ "-nek" == "$1" ]] ; then
  shift
  make $* M="4 8 10 12 16 64 100 144" N="4 8 10 12 16 64 100 144" K="4 8 10 12" \
    BETA=0 OFFLOAD=0 MIC=0 THRESHOLD=$((16*16*256+1))
  elif [[ "-nekbone" == "$1" ]] ; then
  shift
  make $* M="10 16 18 100 256 324" N="10 16 18 100 256 324" K="10 16 18" \
    BETA=0 OFFLOAD=0 MIC=0 THRESHOLD=$((18*18*18*18+1)) 
  elif [[ "-cp2k" == "$1" ]] ; then
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
else
  make -e $*
fi

