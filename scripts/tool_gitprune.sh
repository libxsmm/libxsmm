#!/usr/bin/env bash
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

GIT=$(command -v git)

if [ "${GIT}" ]; then
  ${GIT} reflog expire --expire=now --all
  ${GIT} gc --prune=now
else
  >&2 echo "Error: missing prerequisites!"
  exit 1
fi

