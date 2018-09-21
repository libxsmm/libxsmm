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
#ifndef LIBXSMM_DNN_GRUCELL_H
#define LIBXSMM_DNN_GRUCELL_H

#include "libxsmm_macros.h"
#include "libxsmm_typedefs.h"
#include "libxsmm_dnn.h"


LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_grucell_desc {
  int N;
  int nThreads;
  int m;     /* number of outputs */
  int n;     /* size of the minibatch */
  int k;     /* number of inputs */
  int t;     /* number of time steps */
  int bm;    /* blocksize for m */
  int bn;    /* blocksize for n */
  int bk;    /* blocksize for k */
  int reuse; /* reuse/overwrite memory for FWD */
  int pass;  /* denotes whether it is FWD/BWD/UPD */
  libxsmm_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxsmm_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxsmm_dnn_tensor_format buffer_format;  /* format which is for buffer buffers */
} libxsmm_dnn_grucell_desc;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_grucell {
  int N;
  int nThreads;
  libxsmm_dnn_grucell_desc desc;
  libxsmm_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxsmm_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxsmm_dnn_tensor_format buffer_format;  /* format which is for buffer buffers */
  libxsmm_blasint m;
  libxsmm_blasint n;
  libxsmm_blasint k;
  libxsmm_blasint t;
  libxsmm_blasint bm;
  libxsmm_blasint bn;
  libxsmm_blasint bk;
  int reuse;
  int pass;
  libxsmm_blasint b_m1;
  libxsmm_blasint b_n1;
  libxsmm_blasint b_k1;
  libxsmm_blasint b_m2;
  libxsmm_blasint b_n2;
  libxsmm_blasint b_k2;
  libxsmm_dnn_tensor* wr;
  libxsmm_dnn_tensor* wz;
  libxsmm_dnn_tensor* wg;
  libxsmm_dnn_tensor* xt;
  libxsmm_dnn_tensor* ur;
  libxsmm_dnn_tensor* uz;
  libxsmm_dnn_tensor* ug;
  libxsmm_dnn_tensor* br;
  libxsmm_dnn_tensor* bz;
  libxsmm_dnn_tensor* bg;
  libxsmm_dnn_tensor* h;
  libxsmm_dnn_tensor* r1t;
  libxsmm_dnn_tensor* r2t;
  libxsmm_dnn_tensor* z1t;
  libxsmm_dnn_tensor* z2t;
  libxsmm_dnn_tensor* g1t;
  libxsmm_dnn_tensor* g2t;
  libxsmm_dnn_tensor* g3;
  libxsmm_dnn_tensor* h1;
  libxsmm_dnn_tensor* h2;
  libxsmm_dnn_tensor* h3;
  libxsmm_dnn_tensor* r;
  libxsmm_dnn_tensor* z;
  libxsmm_dnn_tensor* g;
  libxsmm_dnn_tensor* d3;
  libxsmm_dnn_tensor* d4;
  libxsmm_dnn_tensor* d5;
  libxsmm_dnn_tensor* d6;
  libxsmm_dnn_tensor* d7;
  libxsmm_dnn_tensor* d8;
  libxsmm_dnn_tensor* d9;
  libxsmm_dnn_tensor* d10;
  libxsmm_dnn_tensor* d11;
  libxsmm_dnn_tensor* d12;
  libxsmm_dnn_tensor* d13;
  libxsmm_dnn_tensor* d14;
  libxsmm_dnn_tensor* d15;
  libxsmm_dnn_tensor* d16;
  libxsmm_dnn_tensor* d17;
  libxsmm_dnn_tensor* d18;
  libxsmm_dnn_tensor* d19;
  libxsmm_dnn_tensor* d20;
  libxsmm_dnn_tensor* d21;
  libxsmm_dnn_tensor* d22;
  libxsmm_dnn_tensor* d23;
  libxsmm_dnn_tensor* d10M;
  libxsmm_dnn_tensor* d11M;
  libxsmm_dnn_tensor* d18M;
  libxsmm_dnn_tensor* hrTp;
  libxsmm_dnn_tensor* djdwr;
  libxsmm_dnn_tensor* djdwz;
  libxsmm_dnn_tensor* djdwg;
  libxsmm_dnn_tensor* djdxt;
  libxsmm_dnn_tensor* djdur;
  libxsmm_dnn_tensor* djduz;
  libxsmm_dnn_tensor* djdug;
  libxsmm_dnn_tensor* djdht;
  libxsmm_dnn_tensor* djdbr;
  libxsmm_dnn_tensor* djdbz;
  libxsmm_dnn_tensor* djdbg;
  libxsmm_bgemm_handle* handleux;
  libxsmm_bgemm_handle* handlewh;
  libxsmm_bgemm_handle* handlett;
  libxsmm_bgemm_handle* handlewd;
  libxsmm_barrier* barrier; /* barrier */
} libxsmm_dnn_grucell;

LIBXSMM_API libxsmm_dnn_grucell* libxsmm_dnn_create_grucell(libxsmm_dnn_grucell_desc grucell_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_grucell(const libxsmm_dnn_grucell* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_grucell_create_tensor_datalayout(const libxsmm_dnn_grucell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API size_t libxsmm_dnn_grucell_get_scratch_size(const libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_bind_scratch(libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_release_scratch(libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind);

LIBXSMM_API size_t libxsmm_dnn_grucell_get_internalstate_size(const libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_bind_internalstate(libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind, const void* internalstate);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_release_internalstate(libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_assign_internalstate(libxsmm_dnn_grucell* handle, const void* rgoldtb, const void* zgoldtb, const void* ggoldtb);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_bind_tensor(libxsmm_dnn_grucell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_grucell_get_tensor(libxsmm_dnn_grucell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_release_tensor(libxsmm_dnn_grucell* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_fwd(libxsmm_dnn_grucell* gru, int start_thread, int tid);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_bwd_upd_bu(libxsmm_dnn_grucell* gru, int start_thread, int tid, int pass);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_execute_st(libxsmm_dnn_grucell* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXSMM_DNN_GRUCELL_H*/

