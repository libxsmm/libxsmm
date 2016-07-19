#!/bin/sh

cat <<'EOM'
/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_SOURCE_H
#define LIBXSMM_SOURCE_H

#if !defined(LIBXSMM_API)
# define LIBXSMM_API LIBXSMM_RETARGETABLE
# define LIBXSMM_API_DEFINITION LIBXSMM_INLINE LIBXSMM_RETARGETABLE
#else
# error Please do not include any LIBXSMM header other than libxsmm_source.h!
#endif

/**
 * TODO
 */
#include "libxsmm.h"

#include "libxsmm_timer.h"
EOM

HERE=$(cd $(dirname $0); pwd -P)

for FILE in $(ls -1 ${HERE}/../src/*.h); do
  BASENAME=$(basename ${FILE})
  if [ "" != "$(echo ${BASENAME} | grep -v '.template.')" ]; then
    echo "#include \"../src/${BASENAME}\""
  fi
done

echo

for FILE in $(grep -L "LIBXSMM_BUILD..*LIBXSMM_..*_NOINLINE" ${HERE}/../src/*.h); do
  BASENAME=$(basename ${FILE} .h).c
  if [ -e ${HERE}/../src/${BASENAME} ] && [ "" != "$(echo ${BASENAME} | grep -v '.template.')" ]; then
    echo "#include \"../src/${BASENAME}\""
  fi
done

cat <<'EOM'

#endif /*LIBXSMM_SOURCE_H*/
EOM

