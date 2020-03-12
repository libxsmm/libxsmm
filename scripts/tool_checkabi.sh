#!/usr/bin/env bash
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

HERE=$(cd "$(dirname "$0")"; pwd -P)
LIBS=${HERE}/../lib

#EXCLUDE="libxsmmgen"
INCLUDE="libxsmmf"
#INCLUDE="*"
ABINEW=.abi.log
ABITMP=.abi.tmp
ABICUR=.abi.txt

#LIBARGS="--defined-only"
if [ -e "${LIBS}"/${INCLUDE}.so ]; then
  LIBARGS="${LIBARGS} -D"
  LIBTYPE=so
elif [ -e "${LIBS}"/${INCLUDE}.a ]; then
  LIBTYPE=a
else
  LIBTYPE=lib
fi

BASENAME=$(command -v basename)
SORT=$(command -v sort)
DIFF=$(command -v diff)
SED=$(command -v sed)
CUT=$(command -v cut)
LS=$(command -v ls)
WC=$(command -v wc)
CP=$(command -v cp)
MV=$(command -v mv)
NM=$(command -v nm)

if [ "" != "${NM}"   ] && [ "" != "${SED}"  ] && [ "" != "${CUT}" ] && \
   [ "" != "${LS}"   ] && [ "" != "${CP}"   ] && [ "" != "${MV}"  ] && \
   [ "" != "${WC}" ] && [ "" != "${SORT}" ] && [ "" != "${DIFF}" ];
then
  # determine behavior of sort command
  export LC_ALL=C IFS=$'\n'
  if [ "0" != "$(${LS} -1 "${LIBS}"/${INCLUDE}.${LIBTYPE} 2>/dev/null | ${WC} -l)" ]; then
    ${CP} /dev/null ${ABINEW}
    for LIBFILE in $(${LS} -1 "${LIBS}"/*.${LIBTYPE} 2>/dev/null); do
      LIB=$(${BASENAME} "${LIBFILE}" .${LIBTYPE})
      if [ "" = "${EXCLUDE}" ] || [ "" != "$(echo "${EXCLUDE}" | ${SED} "/\b${LIB}\b/d")" ]; then
        echo "Checking ${LIB}..."
        while read LINE; do
          SYMBOL=$(echo "${LINE}" | ${SED} -n "/ T /p" | ${CUT} -d" " -f3)
          if [ "" != "${SYMBOL}" ]; then
            # cleanup compiler-specific symbols (Intel Fortran, GNU Fortran)
            SYMBOL=$(echo ${SYMBOL} | ${SED} \
              -e "s/^libxsmm_mp_libxsmm_\(..*\)_/libxsmm_\1/" \
              -e "s/^__libxsmm_MOD_libxsmm_/libxsmm_/")
            if [ "" != "$(echo ${SYMBOL} | ${SED} -n "/^libxsmm[^.]/p")" ];
            then
              echo "${SYMBOL}" >> ${ABINEW}
            elif [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^__libxsmm_MOD___/p")" ] && \
                 [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^__wrap_..*/p")" ] && \
                 [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^internal_/p")" ] && \
                 [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^libxsmm._/p")" ] && \
                 [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^.gem._/p")" ] && \
                 [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^memalign/p")" ] && \
                 [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^realloc/p")" ] && \
                 [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^malloc/p")" ] && \
                 [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^free/p")" ] && \
                 [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^_init/p")" ] && \
                 [ "" = "$(echo ${SYMBOL} | ${SED} -n "/^_fini/p")" ];
            then
              echo "Error: non-conforming function name"
              echo "${LIB} -> ${SYMBOL}"
              exit 1
            fi
          else
            LOCATION=$(echo "${LINE}" | ${SED} -n "/..*\.o:$/p")
            if [ "" != "${LOCATION}" ]; then
              OBJECT=$(echo "${LOCATION}" | ${SED} -e "s/:$//")
            fi
          fi
        done < <(${NM} ${LIBARGS} "${LIBFILE}" 2>/dev/null)
      else
        echo "Excluded ${LIB}"
      fi
    done
    ${SORT} -u ${ABINEW} > ${ABITMP}
    ${MV} ${ABITMP} ${ABINEW}
    REMOVED=$(${DIFF} --new-line-format="" --unchanged-line-format="" <(${SORT} ${ABICUR}) ${ABINEW})
    NOTE="Note: LIBXSMM must be built with \"make STATIC=0 SYM|DBG=1\"!"
    if [ "" = "${REMOVED}" ]; then
      ${CP} ${ABINEW} ${ABICUR}
      if [ "so" != "${LIBTYPE}" ]; then
        echo "${NOTE}"
      fi
      echo "Successfully completed."
    else
      echo "Error: removed or renamed function(s)"
      echo "${REMOVED}"
      echo "${NOTE}"
    fi
  elif [ -e "${LIBS}"/${INCLUDE}.${LIBTYPE} ]; then
    echo "Error: ABI checker requires shared libraries (${LIBTYPE})."
  else
    echo "Error: ABI checker requires Fortran interface (${INCLUDE})."
  fi
else
  echo "Error: missing prerequisites!"
  exit 1
fi

