/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/

#include "libxsmm_main.h"

LIBXSMM_API libxsmm_dnn_fusedbn* libxsmm_dnn_create_fusedbn(libxsmm_dnn_fusedbn_desc fusedbn_desc, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_fusedbn* handle = 0;
  int noarch;

  if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)) || 
       ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32))    ) {
    handle = (libxsmm_dnn_fusedbn*)malloc(sizeof(libxsmm_dnn_fusedbn));

    if (0 != handle) {
      *status = LIBXSMM_DNN_SUCCESS;
      /* zero entire content; not only safer but also sets data and code pointers to NULL */
      memset(handle, 0, sizeof(*handle));
      /* let's make the desciption presitent */
      handle->desc = fusedbn_desc;
      /* we need to compute the memory layout given the */
      *status = libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.C, 
                                                    &(handle->ifmblock), &(handle->ifmblock_hp),
                                                    &(handle->ofmblock), &(handle->ofmblock_lp),
                                                    &(handle->fm_lp_block), handle->desc.datatype_in, handle->desc.datatype_out, &noarch );
      /* compute the outer blocks */
      if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
        handle->blocksifm = handle->desc.C / handle->ifmblock_hp;
        handle->blocksofm = handle->desc.C / handle->ofmblock;
        handle->blocksifm_lp = handle->desc.C / handle->ifmblock_hp;
        handle->blocksofm_lp = handle->desc.C / handle->ofmblock;
      } else {
        /* this is FP32 */
        handle->blocksifm = handle->desc.C / handle->ifmblock;
        handle->blocksofm = handle->desc.C / handle->ofmblock;
        handle->blocksifm_lp = handle->blocksifm;
        handle->blocksofm_lp = handle->blocksofm;
      }
      /* create barrier */
      handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_fusedbn(const libxsmm_dnn_fusedbn* handle){
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_fusedbn*)handle);
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_fusedbn_create_tensor_datalayout(const libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  return 0;
}

LIBXSMM_API size_t libxsmm_dnn_fusedbn_get_scratch_size(const libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status) {
  return 0;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_bind_scratch(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_compute_kind kind, const void* scratch) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_release_scratch(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_compute_kind kind) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_bind_tensor(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_fusedbn_get_tensor(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  return 0;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_release_tensor(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_execute_st(libxsmm_dnn_fusedbn* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  return status;
}

