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
#ifndef LIBXSMM_DNN_SOFTMAXLOSS_FORWARD_H
#define LIBXSMM_DNN_SOFTMAXLOSS_FORWARD_H

#include <libxsmm_dnn_softmaxloss.h>

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_st_fwd_ncnc(libxsmm_dnn_softmaxloss* handle, int start_thread, int tid);

#endif /* LIBXSMM_DNN_SOFTMAXLOSS_FORWARD_H */
