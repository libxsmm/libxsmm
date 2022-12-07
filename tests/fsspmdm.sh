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
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd -P)

# adjust test properties
export LIBXSMM_FSSPMDM_HINT=$((RANDOM%3+1))
export TEST_N=48

"${HERE}/../samples/pyfr/test.sh" -o /dev/null -n 5 "$@"
