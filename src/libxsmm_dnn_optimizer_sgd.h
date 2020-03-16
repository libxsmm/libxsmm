/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_OPTIMIZER_SGD_H
#define LIBXSMM_DNN_OPTIMIZER_SGD_H

#include <libxsmm_dnn_optimizer.h>

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_optimizer_sgd_st(libxsmm_dnn_optimizer* handle, int start_thread, int tid);

#endif /* LIBXSMM_DNN_OPTIMIZER_SGD_H */
