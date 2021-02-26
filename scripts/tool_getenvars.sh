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

FIND=$(command -v find)
SORT=$(command -v sort)
SED=$(command -v gsed)

# GNU sed is desired (macOS)
if [ "" = "${SED}" ]; then
  SED=$(command -v sed)
fi

HERE="$(cd "$(dirname "$0")" && pwd -P)"
SRC="${HERE}/../src"
EXT="c"

if [ "${FIND}" ] && [ "${SORT}" ] && [ "${SED}" ] && [ -d ${SRC} ]; then
  export LC_ALL=C
  ENVARS="$(${FIND} ${SRC} -type f -name "*.${EXT}" -exec \
    ${SED} -n "s/.*getenv[[:space:]]*([[:space:]]*\"\(.[^\"]*\)..*/\1/p" {} \; | \
    ${SORT} -u)"
  echo "============================="
  echo "Other environment variables"
  echo "============================="
  echo "${ENVARS}" | ${SED} "/LIBXSMM_/d"
  echo "============================="
  echo "LIBXSMM environment variables"
  echo "============================="
  echo "${ENVARS}" | ${SED} -n "/LIBXSMM_/p"
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi
