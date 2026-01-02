#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################

# sde can be downloaded here
SDE64_BIN=/swtools/sde/kits/latest/sde64
SDE64_ARCH="-skx"
SDE64_FLAGS="-ptr_check -null_check -ptr_raise"
SDE=${SDE64_BIN}" "${SDE64_FLAGS}" "${SDE64_ARCH}" -- "
GREP=$(command -v grep)

# iterations, order, precision, arch  and format
if [ $# -eq 5 ]
then
  REPS=$1
  PDEG=$2
  PREC=$3
  VLEN=$4
  FRMT=$5
else
  REPS=1000
  PDEG=5
  PREC=f64
  VLEN=64
  FRMT=csr
fi

if [[ $PDEG == "1" ]]
then
  K=4
  N=3
elif [[ $PDEG == "2" ]]
then
  K=10
  N=6
elif [[ $PDEG == "3" ]]
then
  K=20
  N=10
elif [[ $PDEG == "4" ]]
then
  K=35
  N=15
elif [[ $PDEG == "5" ]]
then
  K=56
  N=21
elif [[ $PDEG == "6" ]]
then
  K=84
  N=28
else
  echo "PDEG need to be in the range of 1 to 6"
  return -1
fi

if [[ $PREC == "f32" ]]
then
  if [[ $VLEN == "32" ]]
  then
    CRUN=8
  elif [[ $VLEN == "64" ]]
  then
    CRUN=16
  elif [[ $VLEN == "16" ]]
  then
    CRUN=4
  else
    echo "VLEN need to be either 16/32/64"
    return -3
  fi
elif [[ $PREC == "f64" ]]
then
  if [[ $VLEN == "32" ]]
  then
    CRUN=4
  elif [[ $VLEN == "64" ]]
  then
    CRUN=8
  elif [[ $VLEN == "16" ]]
  then
    CRUN=2
  else
    echo "VLEN need to be either 16/32/64"
    return -3
  fi
else
  echo "PREC needs to be either f32/f64"
  return -2
fi

if [[ $VLEN == "64" ]]
then
  #on an AVX512 platform we can run natively
  CPUFLAGS=$(if [ "${GREP}" ] && [ -e /proc/cpuinfo ]; then ${GREP} -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
  if [ "$(echo "${CPUFLAGS}" | ${GREP} -o avx512f)" ]; then
    SDE=
  fi
  if [ "$(echo "${CPUFLAGS}" | ${GREP} -o asimd)" ]; then
    SDE=
  fi
else
  SDE=
fi

# number of quantities is always 9
M=9

if [[ $FRMT == "csr" ]]
then
  # test flux matrices, CSR
  for i in `ls mats/tet4_${PDEG}_fluxN*_csr.mtx`; do ${SDE} ./bsparse_packed_csr_${PREC} ${M} ${N} ${K} ${CRUN} ${REPS} $i; done
  for i in `ls mats/tet4_${PDEG}_fluxT*_csr.mtx`; do ${SDE} ./bsparse_packed_csr_${PREC} ${M} ${K} ${N} ${CRUN} ${REPS} $i; done
  # test stiffness matrices, CSR
  for i in `ls mats/tet4_${PDEG}_stiff*_csr.mtx`; do ${SDE} ./bsparse_packed_csr_${PREC} ${M} ${K} ${K} ${CRUN} ${REPS} $i; done
elif [[ $FRMT == "csc" ]]
then
  # test flux matrices, CSC
  for i in `ls mats/tet4_${PDEG}_fluxN*_csc.mtx`; do ${SDE} ./bsparse_packed_csc_${PREC} ${M} ${N} ${K} ${CRUN} ${REPS} $i; done
  for i in `ls mats/tet4_${PDEG}_fluxT*_csc.mtx`; do ${SDE} ./bsparse_packed_csc_${PREC} ${M} ${K} ${N} ${CRUN} ${REPS} $i; done
  # test stiffness matrices, CSC
  for i in `ls mats/tet4_${PDEG}_stiff*_csc.mtx`; do ${SDE} ./bsparse_packed_csc_${PREC} ${M} ${K} ${K} ${CRUN} ${REPS} $i; done
else
  echo "FRMT need to be either csr/csc"
  return -4
fi
# test star matrices
${SDE} ./asparse_packed_csr_${PREC} ${M} ${K} ${M} ${CRUN} ${REPS} mats/tet4_starMatrix_csr.mtx
# test flux matrices
${SDE} ./asparse_packed_csr_${PREC} ${M} ${K} ${M} ${CRUN} ${REPS} mats/tet4_fluxMatrix_csr_sp.mtx
${SDE} ./asparse_packed_csr_${PREC} ${M} ${K} ${M} ${CRUN} ${REPS} mats/tet4_fluxMatrix_csr_de.mtx

