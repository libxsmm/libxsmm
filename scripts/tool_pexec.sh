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
set -o pipefail

BASENAME=$(command -v basename)
XARGS=$(command -v xargs)
SED=$(command -v gsed)

HERE=$(cd "$(dirname "$0")" && pwd -P)
CPU=${HERE}/tool_cpuinfo.sh

# GNU sed is desired (macOS)
if [ "" = "${SED}" ]; then
  SED=$(command -v sed)
fi

if [ "${BASENAME}" ] && [ "${XARGS}" ] && [ "${SED}" ] && [ -e "${CPU}" ]; then
  NC=$1
  if [ ! "${NC}" ] || [ "0" = "${NC}" ]; then
    HERE=$(cd "$(dirname "$0")" && pwd -P)
    NC=$(${CPU} -nc)
  fi
  ${XARGS} -I{} -P "${NC}" bash -c "{} || ( \
    1>&2 echo 'ERROR: {}' && exit 255)" < /dev/stdin 2> >( \
    ${SED} "/xargs/d" >&2)
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi
