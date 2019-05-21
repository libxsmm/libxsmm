#!/bin/sh
#############################################################################
# Copyright (c) 2019, Intel Corporation                                     #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions        #
# are met:                                                                  #
# 1. Redistributions of source code must retain the above copyright         #
#    notice, this list of conditions and the following disclaimer.          #
# 2. Redistributions in binary form must reproduce the above copyright      #
#    notice, this list of conditions and the following disclaimer in the    #
#    documentation and/or other materials provided with the distribution.   #
# 3. Neither the name of the copyright holder nor the names of its          #
#    contributors may be used to endorse or promote products derived        #
#    from this software without specific prior written permission.          #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    #
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  #
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#############################################################################
# Hans Pabst (Intel Corp.)
#############################################################################

HERE=$(cd $(dirname $0); pwd -P)

MKTEMP=${HERE}/../../.mktmp.sh
SED=$(command -v sed)
CP=$(command -v cp)
RM=$(command -v rm)

CLEANUP="-o -D"
JOBDIR=kernel_test
JOBEXT=slurm

if [ "" != "${MKTEMP}" ] && [ "" != "${SED}" ] && \
   [ "" != "${CP}" ] && [ "" != "${RM}" ];
then
  # remove any leftover temporary files
  ${RM} -f .${JOBDIR}_??????.${JOBEXT}
  # create temporary file to avoid sed's i-flag
  JOBTMPFILE=$(${MKTEMP} ${HERE}/.${JOBDIR}_XXXXXX.${JOBEXT})
  # disable glob in Shell
  #set -f
  for CLEAN in ${CLEANUP}; do
    CLEAN_CHECK="${CLEAN_CHECK}/^#SBATCH[[:space:]][[:space:]]*${CLEAN}[[:space:]][[:space:]]*/p;"
    CLEAN_CLEAN="${CLEAN_CLEAN}/^#SBATCH[[:space:]][[:space:]]*${CLEAN}[[:space:]][[:space:]]*/d;"
  done
  COUNT_TOTAL=0
  COUNT_CLEAN=0
  for JOBFILE in $(ls -1 ${HERE}/${JOBDIR}/*.${JOBEXT}); do
    if [ "" != "$(${SED} -n "${CLEAN_CHECK}" ${JOBFILE})" ];
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
  echo "Error: missing prerequisites!"
  exit 1
fi

