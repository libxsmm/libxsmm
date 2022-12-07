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
EXEC=${HERE}/../scripts/tool_pexec.sh

cd "${HERE}/../samples/utilities/wrap" && cat <<EOM | ${EXEC} -o /dev/null -c 3- "$@"
./wrap-test.sh dgemm_batch_strided 100
./wrap-test.sh dgemm_batch_strided 35 16 20 35 35 16    0    0   0 1024  1 0.0 100
./wrap-test.sh dgemm_batch_strided 20 20 32 24 32 24 1000 1000 500 2000  1 0.0 100
./wrap-test.sh dgemm_batch_strided 24 23 21 32 32 32    0    0   0  999 -1 0.5 100
./wrap-test.sh dgemm_batch 100
./wrap-test.sh dgemm_batch 35 16 20 35 35 16 1024  1 0.0 100
./wrap-test.sh dgemm_batch 20 20 32 24 32 24 2000  1 0.0 100
./wrap-test.sh dgemm_batch 24 23 21 32 32 32  999 -1 0.5 100
./wrap-test.sh dgemm 1000
./wrap-test.sh dgemm 350  16  20 350  35 350  1 0.0 1000
./wrap-test.sh dgemm  24  23  21  32  32  32 -1 0.5 1000
./wrap-test.sh dgemm 200 200 200 256 256 256  1 0.0 10
./wrap-test.sh dgemv 10000
./wrap-test.sh dgemv 350  20 350 1 1 1 0 10000
./wrap-test.sh dgemv 200 200 256 1 1 1 0 10000
./wrap-test.sh dgemv  24  21  32 2 2 1 1 10000
EOM
