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

cd ${HERE}/../samples/utilities/dispatch
cat <<EOM | ${EXEC} -o /dev/null "$@"
./dispatch $((SIZE*1)) 1
./dispatch $((SIZE*2)) 1
./dispatch $((SIZE*3)) 1
./dispatch $((SIZE*1)) 2
./dispatch $((SIZE*2)) 2
./dispatch $((SIZE*3)) 2
./dispatch $((SIZE*1)) 3
./dispatch $((SIZE*2)) 3
./dispatch $((SIZE*3)) 3
./dispatch $((SIZE*1)) 4
./dispatch $((SIZE*2)) 4
./dispatch $((SIZE*3)) 4
./dispatch $((SIZE*1)) 7
./dispatch $((SIZE*2)) 7
./dispatch $((SIZE*3)) 7
./dispatch $((SIZE*1)) 8
./dispatch $((SIZE*2)) 8
./dispatch $((SIZE*3)) 8
EOM
