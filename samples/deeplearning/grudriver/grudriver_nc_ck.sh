#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.), Kunal Banerjee (Intel Corp.)
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)
NAME=$(basename $0 .sh)
GREP=$(command -v grep)
ENV=$(command -v env)

if [ "Windows_NT" = "${OS}" ]; then
  # Cygwin's "env" does not set PATH ("Files/Black: No such file or directory")
  export PATH=${PATH}:${HERE}/../../lib:/usr/x86_64-w64-mingw32/sys-root/mingw/bin
  # Cygwin's ldd hangs with dyn. linked executables or certain shared libraries
  LDD=$(command -v cygcheck)
  EXE=.exe
else
  if [ "$(command -v ldd)" ]; then
    LDD=ldd
  elif [ "$(command -v otool)" ]; then
    LDD="otool -L"
  else
    LDD=echo
  fi
fi

MICINFO=$(command -v micinfo)
if [ "${MICINFO}" ]; then
  MICCORES=$(${MICINFO} 2>/dev/null | sed -n "0,/[[:space:]]\+Total No of Active Cores :[[:space:]]\+\([0-9]\+\)/s//\1/p")
fi
if [ "" = "${MICCORES}" ]; then
  MICCORES=61
fi
MICTPERC=3

if [ "-mic" != "$1" ]; then
  if [ "$(${LDD} ${HERE}/${NAME}${EXE} 2>/dev/null | ${GREP} libiomp5\.)" ]; then
    ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../../lib \
      DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../../lib \
      KMP_AFFINITY=compact,granularity=fine,1 \
      MIC_KMP_AFFINITY=compact,granularity=fine \
      MIC_KMP_HW_SUBSET=$((MICCORES-1))c${MICTPERC}t \
      MIC_ENV_PREFIX=MIC \
      OFFLOAD_INIT=on_start \
    ${TOOL_COMMAND} ${HERE}/${NAME}${EXE} "$@"
  else
    ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../../lib \
      DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../../lib \
      OMP_PROC_BIND=TRUE \
    ${TOOL_COMMAND} ${HERE}/${NAME}${EXE} "$@"
  fi
else
  shift
  ${ENV} \
    SINK_LD_LIBRARY_PATH=${SINK_LD_LIBRARY_PATH}:${MIC_LD_LIBRARY_PATH}:${HERE}/../../lib \
  micnativeloadex \
    ${HERE}/${NAME}${EXE} -a "$*" \
    -e "KMP_AFFINITY=compact,granularity=fine" \
    -e "MIC_KMP_HW_SUBSET=$((MICCORES-1))${MICTPERC}t"
fi

ITERS=100
CHKVAL=1

export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,compact,1,0


echo "GRU FWD"
CHECK=${CHKVAL} ./grudriver_nc_ck  ${ITERS}  0  168   256   256  50  24  64  64
wait
CHECK=${CHKVAL} ./grudriver_nc_ck  ${ITERS}  0  168   512   512  50  24  64  64
wait
CHECK=${CHKVAL} ./grudriver_nc_ck  ${ITERS}  0  168  1024  1024  50  24  64  64
wait
CHECK=${CHKVAL} ./grudriver_nc_ck  ${ITERS}  0  168  2048  2048  50  24  64  64
wait
CHECK=${CHKVAL} ./grudriver_nc_ck  ${ITERS}  0  168  4096  4096  50  24  64  64
wait

echo "GRU BWD+UPD"
CHECK=${CHKVAL} ./grudriver_nc_ck  ${ITERS}  3  168   256   256  50  24  64  64
wait
CHECK=${CHKVAL} ./grudriver_nc_ck  ${ITERS}  3  168   512   512  50  24  64  64
wait
CHECK=${CHKVAL} ./grudriver_nc_ck  ${ITERS}  3  168  1024  1024  50  24  64  64
wait
CHECK=${CHKVAL} ./grudriver_nc_ck  ${ITERS}  3  168  2048  2048  50  24  64  64
wait
CHECK=${CHKVAL} ./grudriver_nc_ck  ${ITERS}  3  168  4096  4096  50  24  64  64
wait

echo "GRU performance done"
echo ""
