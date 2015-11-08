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
    64, \
    78, \
    16 29 55, \
    32 29 55, \
    12, \
    4 5 7 9 13 25 26 28 32 45"
elif [[ "-nek" == "$1" ]] ; then
  shift
  make -e $* M="4 8 10 12 16 64 100 144" N="4 8 10 12 16 64 100 144" K="4 8 10 12" \
    BETA=0 THRESHOLD=$((144*144*12))
elif [[ "-nekbone" == "$1" ]] ; then
  shift
  make -e $* M="10 16 18 100 256 324" N="10 16 18 100 256 324" K="10 16 18" \
    BETA=0 THRESHOLD=$((324*324*18)) 
elif [[ "-ci" == "$1" ]] ; then
  shift
  make -e $* PEDANTIC=1 ROW_MAJOR=1 MNK=" \
    0 8 15, \
    23 24 42"
elif [[ "-cif" == "$1" ]] ; then
  shift
  make -e $* PEDANTIC=1 MNK=" \
    0 8 15, \
    23 24 42"
elif [[ "-cinek" == "$1" ]] ; then
  shift
  make -e $* PEDANTIC=1 M="4 8" N="4 8" K="4 8" 
else
  make -e $*
fi

