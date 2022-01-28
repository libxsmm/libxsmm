/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_RNNCELL_BACKWARD_WEIGHT_UPDATE_H
#define LIBXSMM_DNN_RNNCELL_BACKWARD_WEIGHT_UPDATE_H

#include <libxsmm_dnn.h>
#include <libxsmm_dnn_rnncell.h>

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);

#endif /* LIBXSMM_DNN_RNNCELL_BACKWARD_WEIGHT_UPDATE_H */
