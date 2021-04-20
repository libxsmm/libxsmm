#!/usr/bin/env sh
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/hfp/libxsmm/                        #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)

MKTEMP=${HERE}/../../.mktmp.sh
SED=$(command -v sed)
CP=$(command -v cp)
RM=$(command -v rm)

CLEANUP="-o -D"
JOBDIR=kernel_test
JOBEXT=slurm

if [ "${MKTEMP}" ] && [ "${SED}" ] && \
   [ "${CP}" ] && [ "${RM}" ];
then
  # remove any leftover temporary files
  ${RM} -f .${JOBDIR}_??????.${JOBEXT}
  # create temporary file to avoid sed's i-flag
  JOBTMPFILE=$(${MKTEMP} ${HERE}/.${JOBDIR}_XXXXXX.${JOBEXT})
  # disable glob in Shell
  #set -f
  for CLEAN in ${CLEANUP}; do
    CLEAN_CHECK="${CLEAN_CHECK}/^#SBATCH[[:space:]][[:space:]]*${CLEAN}\([[:space:]=][[:space:]=]*\|$\)/p;"
    CLEAN_CLEAN="${CLEAN_CLEAN}/^#SBATCH[[:space:]][[:space:]]*${CLEAN}\([[:space:]=][[:space:]=]*\|$\)/d;"
  done
  CLEAN_CHECK="${CLEAN_CHECK}/^LIBXSMM_TARGET=/p;"
  CLEAN_CLEAN="${CLEAN_CLEAN}/^LIBXSMM_TARGET=/d;"
  COUNT_TOTAL=0
  COUNT_CLEAN=0
  for JOBFILE in $(ls -1 ${HERE}/${JOBDIR}/*.${JOBEXT}); do
    if [ "$(${SED} -n "${CLEAN_CHECK}" ${JOBFILE})" ];
    then
      echo "Cleaning ${JOBFILE}..."
      ${SED} "${CLEAN_CLEAN}" ${JOBFILE} > ${JOBTMPFILE}
      ${CP} ${JOBTMPFILE} ${JOBFILE}
      COUNT_CLEAN=$((COUNT_CLEAN+1))
    fi
    COUNT_TOTAL=$((COUNT_TOTAL+1))
  done
  ${RM} -f ${JOBTMPFILE}
  if [ "0" != "${COUNT_CLEAN}" ]; then
    echo "Successfully cleaned ${COUNT_CLEAN} of ${COUNT_TOTAL} job files."
  else
    echo "Successfully completed (there was nothing to clean)."
  fi
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi

