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

if [ "${HERE}" ]; then
  "${HERE}"/../samples/edge/test_dense_packedacrm.sh
  "${HERE}"/../samples/edge/test_dense_packedbcrm.sh
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
