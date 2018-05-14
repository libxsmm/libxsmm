/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/

#include <libxsmm.h>
#include "libxsmm_main.h"

LIBXSMM_API libxsmm_dnn_lstmcell* libxsmm_dnn_create_lstmcell(libxsmm_dnn_lstmcell_desc lstmcell_desc, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_lstmcell* handle = 0;

  LIBXSMM_UNUSED( lstmcell_desc );
  *status = LIBXSMM_DNN_SUCCESS;

  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_lstmcell(const libxsmm_dnn_lstmcell* handle) {
  LIBXSMM_UNUSED( handle );

  return LIBXSMM_DNN_SUCCESS;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_lstmcell_create_tensor_datalayout(const libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout = 0;

  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( type );
  *status = LIBXSMM_DNN_SUCCESS;

  return layout;
}


LIBXSMM_API size_t libxsmm_dnn_lstmcell_get_scratch_size(const libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status) {
  size_t size = 0;

  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( kind );
  *status = LIBXSMM_DNN_SUCCESS;

  return size;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_scratch(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, const void* scratch) {
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( kind );
  LIBXSMM_UNUSED( scratch );

  return LIBXSMM_DNN_SUCCESS;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_release_scratch(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind) {
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( kind );

  return LIBXSMM_DNN_SUCCESS;
}


LIBXSMM_API size_t libxsmm_dnn_lstmcell_get_internalstate_size(const libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status) {
  size_t size = 0;

  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( kind );
  *status = LIBXSMM_DNN_SUCCESS;

  return size;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_internalstate(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, const void* internalstate) {
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( kind );
  LIBXSMM_UNUSED( internalstate );

  return LIBXSMM_DNN_SUCCESS;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_release_internalstate(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind) {
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( kind );

  return LIBXSMM_DNN_SUCCESS;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_tensor(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type) {
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( tensor );
  LIBXSMM_UNUSED( type );

  return LIBXSMM_DNN_SUCCESS;
}


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_lstmcell_get_tensor(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor* tensor = 0;

  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( type );
  *status = LIBXSMM_DNN_SUCCESS;

  return tensor;
}

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_release_tensor(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor_type type) {
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( type );

  return LIBXSMM_DNN_SUCCESS;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_execute_st(libxsmm_dnn_lstmcell* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( kind );
  LIBXSMM_UNUSED( start_thread );
  LIBXSMM_UNUSED( tid );

  return LIBXSMM_DNN_SUCCESS;
}

