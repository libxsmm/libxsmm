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
/* Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_LSTMCELL_H
#define LIBXSMM_DNN_LSTMCELL_H

#include "libxsmm_macros.h"
#include "libxsmm_typedefs.h"
#include "libxsmm_dnn.h"


LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_lstmcell_desc {
  int nThreads;
  int K;     /* number of outputs */
  int N;     /* size of the minibatch */
  int C;     /* number of inputs */
  int t;     /* number of time steps */
  int pass;  /* denotes whether it is FWD/BWD/UPD */
  libxsmm_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxsmm_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxsmm_dnn_tensor_format buffer_format;  /* format which is for activation buffers */
  libxsmm_dnn_tensor_format filter_format;  /* format which is for filter buffers */
} libxsmm_dnn_lstmcell_desc;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_lstmcell {
  libxsmm_dnn_lstmcell_desc desc;
  libxsmm_dnn_internal_format custom_format_type; /* required only for comparing layouts  */
  libxsmm_blasint bk;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_dnn_tensor* xt;
  libxsmm_dnn_tensor* csp;
  libxsmm_dnn_tensor* hp;
  libxsmm_dnn_tensor* w;
  libxsmm_dnn_tensor* r;
  libxsmm_dnn_tensor* b;
  libxsmm_dnn_tensor* cst;
  libxsmm_dnn_tensor* ht;
  libxsmm_dnn_tensor* it;
  libxsmm_dnn_tensor* ft;
  libxsmm_dnn_tensor* ot;
  libxsmm_dnn_tensor* cit;
  libxsmm_dnn_tensor* cot;
  libxsmm_dnn_tensor* dxt;
  libxsmm_dnn_tensor* dcspt;
  libxsmm_dnn_tensor* dhpt;
  libxsmm_dnn_tensor* dw;
  libxsmm_dnn_tensor* dr;
  libxsmm_dnn_tensor* db;
  libxsmm_dnn_tensor* dcs;
  libxsmm_dnn_tensor* dht;
  libxsmm_dnn_tensor* dit;
  libxsmm_dnn_tensor* dft;
  libxsmm_dnn_tensor* dot;
  libxsmm_dnn_tensor* dcit;
  libxsmm_dnn_tensor* doutt;
  libxsmm_dnn_tensor* t1;
  libxsmm_dnn_tensor* t2;
  libxsmm_barrier* barrier;
} libxsmm_dnn_lstmcell;

LIBXSMM_API libxsmm_dnn_lstmcell* libxsmm_dnn_create_lstmcell(libxsmm_dnn_lstmcell_desc lstmcell_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_lstmcell(const libxsmm_dnn_lstmcell* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_lstmcell_create_tensor_datalayout(const libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API size_t libxsmm_dnn_lstmcell_get_scratch_size(const libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_scratch(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_release_scratch(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind);

LIBXSMM_API size_t libxsmm_dnn_lstmcell_get_internalstate_size(const libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_internalstate(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, const void* internalstate);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_release_internalstate(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_tensor(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_lstmcell_get_tensor(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_release_tensor(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_fwd(libxsmm_dnn_lstmcell* lstm, int start_thread, int tid);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bwd_upd_bu(libxsmm_dnn_lstmcell* lstm, int start_thread, int tid, int pass);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_execute_st(libxsmm_dnn_lstmcell* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXSMM_DNN_LSTMCELL_H*/

