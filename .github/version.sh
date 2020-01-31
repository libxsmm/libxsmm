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

CUT=$(command -v cut)
GIT=$(command -v git)

NAME=$(${GIT} name-rev --name-only HEAD)
MAIN=$(${GIT} describe --tags --abbrev=0)
REVC=$(${GIT} describe --tags | ${CUT} -d- -f2)

echo "${NAME}-${MAIN}-${REVC}"
