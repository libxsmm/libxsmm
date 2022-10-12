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

GIT=$(command -v git)

if [ "${GIT}" ]; then
  ${GIT} shortlog -sne --all --no-merges
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi

