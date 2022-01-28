#!/usr/bin/env sh
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
MKDIR=$(command -v mkdir)
WGET=$(command -v wget)

NAMES=" \
  MarkupSafe-1.1.1-cp37-cp37m-manylinux1_x86_64.whl \
  translationstring-1.3-py2.py3-none-any.whl \
  itsdangerous-1.1.0-py2.py3-none-any.whl \
  lark_parser-0.8.5-py2.py3-none-any.whl \
  Werkzeug-1.0.1-py2.py3-none-any.whl \
  Jinja2-2.11.2-py2.py3-none-any.whl \
  colander-0.9.9.tar.gz \
  iso8601-0.1.8.tar.gz \
  Flask-0.10.1.tar.gz \
  redis-2.8.0.tar.gz \
  lark-0.0.4.tar.gz"

if [ "${MKDIR}" ] && [ "${WGET}" ]; then
  ${MKDIR} -p ${HERE}/roc/tmp1; cd ${HERE}/roc/tmp1
  for NAME in ${NAMES}; do
    ${WGET} -N https://github.com/libxsmm/libxsmm/raw/master/samples/deeplearning/sparse_training/dlrm/roc/tmp1/${NAME}
  done
fi
