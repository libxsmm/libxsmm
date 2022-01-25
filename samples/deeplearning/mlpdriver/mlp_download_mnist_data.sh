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

HERE=$(cd "$(dirname "$0")" && pwd -P)
MKDIR=$(command -v mkdir)
WGET=$(command -v wget)

# ls -1 | xargs
NAMES="t10k-images.idx3-ubyte t10k-labels.idx1-ubyte train-images.idx3-ubyte train-labels.idx1-ubyte"

if [ "${MKDIR}" ] && [ "${WGET}" ]; then
  ${MKDIR} -p ${HERE}/mnist_data; cd ${HERE}/mnist_data
  for NAME in ${NAMES}; do
    ${WGET} -N https://github.com/hfp/libxsmm/raw/master/samples/deeplearning/mlpdriver/mnist_data/${NAME}
  done
fi

