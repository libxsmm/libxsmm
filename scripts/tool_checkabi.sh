#!/bin/bash
#############################################################################
# Copyright (c) 2018-2019, Intel Corporation                                #
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

#EXCLUDE="libxsmmgen"
ABINEW=.abi.log
ABITMP=.abi.tmp
ABICUR=.abi.txt
LIBTYPE=so

BASENAME=$(command -v basename)
ECHO=$(command -v echo)
SORT=$(command -v sort)
DIFF=$(command -v diff)
SED=$(command -v sed)
CUT=$(command -v cut)
LS=$(command -v ls)
CP=$(command -v cp)
MV=$(command -v mv)
NM=$(command -v nm)

if [ "" != "${ECHO}" ] && [ "" != "${SORT}" ] && [ "" != "${DIFF}" ] && \
   [ "" != "${NM}"   ] && [ "" != "${SED}"  ] && [ "" != "${CUT}"  ] && \
   [ "" != "${LS}"   ] && [ "" != "${CP}"   ] && [ "" != "${MV}"   ];
then
  # determine behavior of sort command
  export LC_ALL=C
  for LIBFILE in $(${LS} -1 ${LIBS}/*.${LIBTYPE} 2>/dev/null); do
    LIBFILES="${LIBFILES} ${LIBFILE}"
  done
  if [ "" != "${LIBFILES}" ]; then
    ${CP} /dev/null ${ABINEW}
    for LIBFILE in ${LIBFILES}; do
      LIB=$(${BASENAME} ${LIBFILE} .${LIBTYPE})
      if [ "" = "${EXCLUDE}" ] || [ "" != "$(${ECHO} "${EXCLUDE}" | ${SED} "/\b${LIB}\b/d")" ]; then
        ${ECHO} "Checking ${LIB}..."
        while read LINE; do
          SYMBOL=$(${ECHO} "${LINE}" | ${SED} -n "/ T /p" | ${CUT} -d" " -f3)
          if [ "" != "${SYMBOL}" ]; then
            if [ "" != "$(${ECHO} ${SYMBOL} | ${SED} -n "/^__libxsmm_MOD_libxsmm/p")" ] || \
               [ "" != "$(${ECHO} ${SYMBOL} | ${SED} -n "/^libxsmm/p")" ];
            then
              ${ECHO} "${SYMBOL}" >> ${ABINEW}
            elif [ "" = "$(${ECHO} ${SYMBOL} | ${SED} -n "/^__libxsmm_MOD___/p")" ] && \
                 [ "" = "$(${ECHO} ${SYMBOL} | ${SED} -n "/^__wrap_..*_/p")" ] && \
                 [ "" = "$(${ECHO} ${SYMBOL} | ${SED} -n "/^.gem._/p")" ] && \
                 [ "" = "$(${ECHO} ${SYMBOL} | ${SED} -n "/^_init/p")" ] && \
                 [ "" = "$(${ECHO} ${SYMBOL} | ${SED} -n "/^_fini/p")" ];
            then
              ${ECHO} "Error: non-conforming function name"
              ${ECHO} "${LIB} -> ${SYMBOL}"
              exit 1
            fi
          else
            LOCATION=$(${ECHO} "${LINE}" | ${SED} -n "/..*\.o:$/p")
            if [ "" != "${LOCATION}" ]; then
              OBJECT=$(${ECHO} "${LOCATION}" | ${SED} -e "s/:$//")
            fi
          fi
        done < <(${NM} ${LIBFILE})
      else
        ${ECHO} "Excluded ${LIB}"
      fi
    done
    ${SORT} -u ${ABINEW} > ${ABITMP}
    ${MV} ${ABITMP} ${ABINEW}
    REMOVED=$(${DIFF} --new-line-format="" --unchanged-line-format="" <(${SORT} ${ABICUR}) ${ABINEW})
    if [ "" = "${REMOVED}" ]; then
      ${CP} ${ABINEW} ${ABICUR}
      ${ECHO} "Successfully completed."
    else
      ${ECHO} "Error: removed or renamed function(s)"
      ${ECHO} "${REMOVED}"
    fi
  else
    ${ECHO} "Warning: ABI checker requires shared libraries (${LIBTYPE})."
  fi
else
  ${ECHO} "Error: missing prerequisites!"
  exit 1
fi

