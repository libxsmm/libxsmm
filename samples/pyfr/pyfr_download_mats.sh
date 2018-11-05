#!/bin/sh
#############################################################################
# Copyright (c) 2017-2018, Intel Corporation                                #
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
MKDIR=$(which mkdir)
WGET=$(which wget)

DATASET="p1 p2 p3 p4 p5 p6"
KINDS="hex pri quad tet tri"
FILES="m0-de m0-sp m132-de m132-sp m3-de m3-sp m460-de m460-sp m6-de m6-sp"

if [ "" != "${MKDIR}" ] && [ "" != "${WGET}" ]; then
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

