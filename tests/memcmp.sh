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
SIZE=1000

export CHECK=1

cd ${HERE}/../samples/utilities/memcmp
cat <<EOM | ${EXEC} -o /dev/null "$@"
./memcmp 0 0 $((SIZE*1)) 0
./memcmp 0 0 $((SIZE*2)) 0
./memcmp 0 0 $((SIZE*3)) 0
./memcmp 0 0 $((SIZE*1)) 1
./memcmp 0 0 $((SIZE*2)) 1
./memcmp 0 0 $((SIZE*3)) 1
./memcmp 0 0 $((SIZE*1)) 2
./memcmp 0 0 $((SIZE*2)) 2
./memcmp 0 0 $((SIZE*3)) 2
./memcmp 0 0 $((SIZE*1)) 4
./memcmp 0 0 $((SIZE*2)) 4
./memcmp 0 0 $((SIZE*3)) 4
./memcmp 0 0 $((SIZE*1)) 8
./memcmp 0 0 $((SIZE*2)) 8
./memcmp 0 0 $((SIZE*3)) 8
./memcmp 0 0 $((SIZE*1)) 16
./memcmp 0 0 $((SIZE*2)) 16
./memcmp 0 0 $((SIZE*3)) 16
./memcmp 0 0 $((SIZE*1)) 17
./memcmp 0 0 $((SIZE*2)) 17
./memcmp 0 0 $((SIZE*3)) 17
./memcmp 0 0 $((SIZE*1)) 23
./memcmp 0 0 $((SIZE*2)) 23
./memcmp 0 0 $((SIZE*3)) 23
./memcmp 0 0 $((SIZE*1)) 32
./memcmp 0 0 $((SIZE*2)) 32
./memcmp 0 0 $((SIZE*3)) 32
EOM
