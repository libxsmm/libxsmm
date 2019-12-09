#!/bin/bash
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

MKTEMP=$(command -v mktemp)
ECHO=$(command -v echo)
MV=$(command -v mv)

if [ "" != "${MKTEMP}" ] && [ "" != "${ECHO}" ] && [ "" != "${MV}" ]; then
  TEMPLATE=${1/XXXXXX/}.XXXXXX
  TMPFILE=$(${MKTEMP} ${TEMPLATE})
  EXTFILE=${TMPFILE: -6}
  NEWFILE=${1/XXXXXX/${EXTFILE}}
  ${MV} ${TMPFILE} ${NEWFILE}
  ${ECHO} "${NEWFILE}"
else
  touch $1
fi

