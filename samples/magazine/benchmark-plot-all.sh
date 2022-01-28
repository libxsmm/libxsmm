#!/usr/bin/env bash
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Hans Pabst (Intel Corp.)
###############################################################################

HERE=$(cd "$(dirname "$0")" && pwd -P)

"${HERE}"/benchmark-plot.sh eigen "$@"
"${HERE}"/benchmark-plot.sh blaze "$@"
"${HERE}"/benchmark-plot.sh xsmm "$@"
"${HERE}"/benchmark-plot.sh xbat "$@"
"${HERE}"/benchmark-plot.sh blas "$@"

