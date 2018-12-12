#!/bin/bash
#############################################################################
# Copyright (c) 2018, Intel Corporation                                     #
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
LIBS=${HERE}/../lib

BASENAME=$(command -v basename 2>/dev/null)
ECHO=$(command -v echo 2>/dev/null)
SED=$(command -v sed 2>/dev/null)
CAT=$(command -v cat 2>/dev/null)
CUT=$(command -v cut 2>/dev/null)
LS=$(command -v ls 2>/dev/null)
RM=$(command -v rm 2>/dev/null)
NM=$(command -v nm 2>/dev/null)

if [ "" != "${ECHO}" ] && [ "" != "${SED}" ] && [ "" != "${CAT}" ] && [ "" != "${CUT}" ] && \
   [ "" != "${LS}" ] && [ "" != "${RM}" ] && [ "" != "${NM}" ];
then
  ${RM} -f .checkabi.log
  for LIBTYPE in a so; do
    for LIBFILE in $(ls -1 ${LIBS}/*.${LIBTYPE} 2>/dev/null); do
      LIB=$(${BASENAME} ${LIBFILE} .${LIBTYPE})
      ${ECHO} "Checking ${LIB}..."
      ${NM} ${LIBFILE} | while read LINE;
      do
        SYMBOL=$(${ECHO} "${LINE}" | ${SED} -n "/ T /p" | ${CUT} -d" " -f3)
        if [ "" != "${SYMBOL}" ]; then
          if [ "" = "$(${ECHO} ${SYMBOL} | ${SED} -n /^libxsmm/p)" ] && \
             [ "" = "$(${ECHO} ${SYMBOL} | ${SED} -n /^_init/p)" ] && \
             [ "" = "$(${ECHO} ${SYMBOL} | ${SED} -n /^_fini/p)" ] && \
             [ "" = "$(${ECHO} ${SYMBOL} | ${SED} -n /^__/p)" ] && \
             [ "" = "$(${ECHO} ${SYMBOL} | ${SED} -n /_$/p)" ];
          then
            ${ECHO} "  ${LIB}.${LIBTYPE} -> ${OBJECT} -> ${SYMBOL}" >> .checkabi.log
          fi
        else
          LOCATION=$(${ECHO} "${LINE}" | ${SED} -n "/..*\.o:$/p")
          if [ "" != "${LOCATION}" ]; then
            OBJECT=$(${ECHO} "${LOCATION}" | ${SED} -e "s/:$//")
          fi
        fi
      done
    done
  done
  if [ -e .checkabi.log ]; then
    ${ECHO} "There are non-conforming function names:"
    ${CAT} .checkabi.log
  else
    ${ECHO} "Successfully Completed."
  fi
else
  ${ECHO} "Error: missing prerequisites!"
  exit 1
fi

