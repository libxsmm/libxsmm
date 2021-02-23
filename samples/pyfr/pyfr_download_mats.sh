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
MKDIR=$(command -v mkdir)
WGET=$(command -v wget)

DATASET="p1 p2 p3 p4 p5 p6"
KINDS="hex pri quad tet tri"
FILES="m0-de m0-sp m132-de m132-sp m3-de m3-sp m460-de m460-sp m6-de m6-sp"

if [ "${MKDIR}" ] && [ "${WGET}" ]; then
  ${MKDIR} -p ${HERE}/mats; cd ${HERE}/mats
  for DATA in ${DATASET}; do
    mkdir ${DATA}; cd ${DATA}
    for KIND in ${KINDS}; do
      mkdir ${KIND}; cd ${KIND}
      for FILE in ${FILES}; do
        ${WGET} -N https://github.com/hfp/libxsmm/raw/master/samples/pyfr/mats/${DATA}/${KIND}/${FILE}.mtx
      done
      cd ..
    done
    cd ..
  done
fi

