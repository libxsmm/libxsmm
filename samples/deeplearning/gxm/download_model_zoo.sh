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

# ls -1 | xargs
NAMES=" \
  cifar10/mean.binaryproto cifar10/solver.prototxt cifar10/train_val.prototxt \
  convnet/solver.prototxt convnet/solver_dump_formula.prototxt convnet/solver_formula.prototxt convnet/test.prototxt convnet/train_val.prototxt convnet/train_val_dump.prototxt convnet/train_val_lmdb.prototxt convnet/train_val_mol.prototxt convnet/train_val_single.prototxt convnet/train_val_split_lmdb.prototxt convnet/xsmm_train_val.prototxt convnet/xsmm_tv_lmdb.prototxt \
  googlenet/v1/solver_1024.prototxt googlenet/v1/solver_256.prototxt googlenet/v1/solver_googlenet_1024.prototxt googlenet/v1/solver_googlenet_1536.prototxt googlenet/v1/train_val_f1k_lrn.prototxt googlenet/v1/train_val_flat.prototxt googlenet/v1/train_val_flat_1024.prototxt googlenet/v1/train_val_flat_full_1024.prototxt googlenet/v1/train_val_lmdb.prototxt googlenet/v1/train_val_mf1k.prototxt googlenet/v1/train_val_sf1k.prototxt googlenet/v1/xsmm_tvf_1024.prototxt googlenet/v1/xsmm_tvf_256.prototxt googlenet/v1/xsmm_tvf_test_1024.prototxt \
  googlenet/v3/dummy_train_val_56.prototxt googlenet/v3/mn_train_val.prototxt googlenet/v3/solver.prototxt googlenet/v3/solver_864.prototxt googlenet/v3/solver_896.prototxt googlenet/v3/train_val.prototxt \
  mnist/lenet_solver.prototxt mnist/lenet_train_val.prototxt \
  resnet/1_resnet50_dummy_24_f32.prototxt resnet/1_resnet50_dummy_26_f32.prototxt resnet/1_resnet50_dummy_28_bf16.prototxt resnet/1_resnet50_dummy_28_f32.prototxt resnet/1_resnet50_dummy_56_bf16.prototxt resnet/1_resnet50_dummy_56_f32.prototxt resnet/1_resnet50_dummy_56_fcbn_bf16.prototxt resnet/1_resnet50_dummy_56_fcbn_f32.prototxt resnet/1_resnet50_dummy_70.prototxt resnet/1_resnet50_dummy_96_f32.prototxt resnet/1_resnet50_image_50_bf16_test.prototxt resnet/1_resnet50_image_50_f32_test.prototxt resnet/1_resnet50_image_50_fcbn_bf16_test.prototxt resnet/1_resnet50_image_56_bf16.prototxt resnet/1_resnet50_image_56_f32.prototxt resnet/1_resnet50_image_56_fcbn_bf16.prototxt resnet/1_resnet50_image_56_fcbn_f32.prototxt resnet/1_resnet50_image_56_test.prototxt resnet/1_resnet50_image_64.prototxt resnet/1_resnet50_image_70.prototxt resnet/1_resnet50_lmdb_56.prototxt resnet/1_resnet50_lmdb_64.prototxt resnet/1_resnet50_lmdb_70.prototxt resnet/1_resnet50_v1p5_dummy_27_f32.prototxt resnet/1_resnet50_v1p5_dummy_27_f32_lp.prototxt resnet/1_resnet50_v1p5_dummy_28_f32.prototxt resnet/1_resnet50_v1p5_dummy_28_f32_lp.prototxt resnet/dummy_solver_864.prototxt resnet/dummy_solver_864_bf16.prototxt resnet/mn_resnet50_dummy_56_bf16.prototxt resnet/mn_resnet50_dummy_56_f32.prototxt resnet/mn_resnet50_dummy_56_fcbn_bf16.prototxt resnet/mn_resnet50_dummy_56_fcbn_f32.prototxt resnet/mn_resnet50_image_56.prototxt resnet/mn_resnet50_image_56_bf16.prototxt resnet/mn_resnet50_image_56_f32.prototxt resnet/mn_resnet50_image_56_fcbn_bf16.prototxt resnet/mn_resnet50_image_56_fcbn_f32.prototxt resnet/solver.prototxt resnet/solver_864.prototxt resnet/solver_864_bf16.prototxt resnet/solver_896.prototxt resnet/solver_896_bf16.prototxt resnet/solver_bf16.prototxt"

if [ "${MKDIR}" ] && [ "${WGET}" ]; then
  for NAME in ${NAMES}; do
    DIR=$(dirname ${NAME})
    ${MKDIR} -p ${HERE}/model_zoo/${DIR}
    cd ${HERE}/model_zoo/${DIR}
    ${WGET} -N https://github.com/libxsmm/libxsmm/raw/master/samples/deeplearning/gxm/model_zoo/${NAME}
  done
fi
