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

${HERE}/../samples/utilities/wrap/wrap-test.sh dgemm_batch
${HERE}/../samples/utilities/wrap/wrap-test.sh dgemm_batch 35 16 20 35 35 16 1024  1 0.0
${HERE}/../samples/utilities/wrap/wrap-test.sh dgemm_batch 20 20 32 24 32 24 2000  1 0.0
${HERE}/../samples/utilities/wrap/wrap-test.sh dgemm_batch 24 23 21 32 32 32  999 -1 0.5

${HERE}/../samples/utilities/wrap/wrap-test.sh dgemm
${HERE}/../samples/utilities/wrap/wrap-test.sh dgemm 350  16  20 350  35 350  1 0.0
${HERE}/../samples/utilities/wrap/wrap-test.sh dgemm 200 200 200 256 256 256  1 0.0
${HERE}/../samples/utilities/wrap/wrap-test.sh dgemm  24  23  21  32  32  32 -1 0.5

${HERE}/../samples/utilities/wrap/wrap-test.sh dgemv
${HERE}/../samples/utilities/wrap/wrap-test.sh dgemv 350  20 350 1 1 1 0
${HERE}/../samples/utilities/wrap/wrap-test.sh dgemv 200 200 256 1 1 1 0
${HERE}/../samples/utilities/wrap/wrap-test.sh dgemv  24  21  32 2 2 1 1
