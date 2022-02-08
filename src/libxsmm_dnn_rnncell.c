/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas, Kunal Banerjee (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_rnncell_forward.h"
#include "libxsmm_dnn_rnncell_backward_weight_update.h"
#include "libxsmm_dnn_elementwise.h"
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

LIBXSMM_API libxsmm_dnn_rnncell* libxsmm_dnn_create_rnncell(libxsmm_dnn_rnncell_desc rnncell_desc, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_rnncell* handle = 0;

  /* init libxsmm */
  LIBXSMM_INIT

  /* some check we can do before allocating the handle */
  if ( (rnncell_desc.datatype_in != rnncell_desc.datatype_out) ||
       ( (rnncell_desc.datatype_in != LIBXSMM_DNN_DATATYPE_BF16) && (rnncell_desc.datatype_in != LIBXSMM_DNN_DATATYPE_F32) ) ) {
    *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
    return NULL;
  }
  /* let's do some simple checks for BF16 as this limits the cell and architecture */
  if ( (rnncell_desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) || (rnncell_desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
    if ( (LIBXSMM_X86_AVX512_CORE > libxsmm_target_archid) || (rnncell_desc.C % 16 != 0) || (rnncell_desc.K % 16 != 0) ) {
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return NULL;
    }
  }
  /* we need at least one timestep */
  if (rnncell_desc.max_T < 1) {
    *status = LIBXSMM_DNN_ERR_TIME_STEPS_TOO_SMALL;
    return NULL;
  }

  /* zero entire content; not only safer but also sets data and code pointers to NULL */
  handle = (libxsmm_dnn_rnncell*)calloc(1, sizeof(libxsmm_dnn_rnncell));
  if (NULL != handle) {
    *status = LIBXSMM_DNN_SUCCESS;
    /* initialize known handle components */
    handle->desc = rnncell_desc;
  /* set current seq length to max length */
    handle->T = rnncell_desc.max_T;
    /* set blocking factors */
    handle->bk = (handle->desc.bk == 0) ? 64 : handle->desc.bk;
    handle->bn = (handle->desc.bn == 0) ? 64 : handle->desc.bn;
    handle->bc = (handle->desc.bc == 0) ? 64 : handle->desc.bc;
    handle->use_fwd_fused_impl = handle->desc.use_fwd_fused_impl;
    handle->fwd_block = handle->desc.fwd_block;
    handle->bwdupd_block = handle->desc.bwdupd_block;
    if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
      handle->lpb = 2;
    } else {
      handle->lpb = 1;
    }
   /* validate blocking factors */
    if ( handle->desc.N % handle->bn != 0 ) {
      handle->bn = handle->desc.N;
      *status = LIBXSMM_DNN_WARN_RNN_SUBOPTIMAL_N_BLOCKING;
    }
    if ( handle->desc.C % handle->bc != 0 ) {
      handle->bc = handle->desc.C;
      *status = LIBXSMM_DNN_WARN_RNN_SUBOPTIMAL_C_BLOCKING;
    }
    if ( handle->desc.K % handle->bk != 0 ) {
      handle->bk = handle->desc.K;
      *status = LIBXSMM_DNN_WARN_RNN_SUBOPTIMAL_K_BLOCKING;
    }

    /* If in SPR, generate tilerelease kernel */
    if ((libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT)) {
      int l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
      handle->tilerelease_kernel = libxsmm_bsmmdispatch(handle->bk, handle->bk, handle->bk, NULL, NULL, NULL, NULL, NULL, &l_tr_flags, NULL);
    }

    /* In case of BF16 for now hoist the BRGEMM and make them to use STRIDED variant by default */
    if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
      libxsmm_blasint BF, CB_BLOCKS, KB_BLOCKS;
      const libxsmm_blasint K =  handle->desc.K;
      const libxsmm_blasint N =  handle->desc.N;
      const libxsmm_blasint C =  handle->desc.C;
      const libxsmm_blasint bk = handle->bk;
      const libxsmm_blasint bn = handle->bn;
      const libxsmm_blasint bc = handle->bc;
      const libxsmm_blasint cBlocks = C/bc;
      const libxsmm_blasint kBlocks = K/bk;
      const libxsmm_blasint nBlocks = N/bn;
      int tc_flags = 0;
      int kernel_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
      size_t stride_a, stride_b;

      if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_SPR) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT)) {
        kernel_flags = ((handle->bk % 32 == 0) && (handle->bc % 32 == 0) && (handle->bn % 32 == 0)) ? LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG : 0;
        kernel_flags = kernel_flags | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
        tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
      }

      /* Blocking reduction domain if it is too large */
      BF = 1;
      if ((C > 1024 && C <= 2048) || (K > 1024 && K <= 2048)) {
        BF = 8;
        while ( (cBlocks % BF != 0) || (kBlocks % BF != 0) ) {
          BF--;
        }
      }
      if (C > 2048 || K > 2048) {
        BF = 16;
        while ( (cBlocks % BF != 0) || (kBlocks % BF != 0) ) {
          BF--;
        }
      }
      if (C == 2048 && K == 1024) {
        BF = 2;
      }
      BF = handle->fwd_block;

      if (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) {
        CB_BLOCKS = cBlocks/BF;
        KB_BLOCKS = kBlocks/BF;

        /* define batch-reduce gemm kernels */
        stride_a = bc * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bc * bn * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->fwd_kernela = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bk, bn, bc, stride_a, stride_b, CB_BLOCKS, &bk, &bc, &bk, NULL, NULL, &kernel_flags, NULL );
        stride_a = bk * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bk * bn * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->fwd_kernelb = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bk, bn, bk, stride_a, stride_b, KB_BLOCKS, &bk, &bk, &bk, NULL, NULL, &kernel_flags, NULL );
        if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_SPR) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT)) {
          handle->fwd_tileconfig = libxsmm_bsmmdispatch_reducebatch_addr( bk, bn, bk, &bk, &K, &K, NULL, NULL, &tc_flags, NULL );
        }

        BF = handle->bwdupd_block;
        KB_BLOCKS = kBlocks/BF;

        stride_a = bc * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bk * bn * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->bwdupd_kernela = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bc, bn, bk, stride_a, stride_b, KB_BLOCKS, &bc, &bk, &bc, NULL, NULL, &kernel_flags, NULL);
        stride_a = bn * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bn * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->bwdupd_kernelb = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bk, bk, bn, stride_a, stride_b, nBlocks, &bk, &bn, &bk, NULL, NULL, &kernel_flags, NULL);
        stride_a = bn * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bn * bc * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->bwdupd_kernelc = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bk, bc, bn, stride_a, stride_b, nBlocks, &bk, &bn, &bk, NULL, NULL, &kernel_flags, NULL);
        stride_a = bk * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bn * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->bwdupd_kerneld = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bk, bn, bk, stride_a, stride_b, KB_BLOCKS, &bk, &bk, &bk, NULL, NULL, &kernel_flags, NULL);
        if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_SPR) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT)) {
          handle->bwdupd_tileconfig = libxsmm_bsmmdispatch_reducebatch_addr( bk, bn, bk, &bk, &K, &K, NULL, NULL, &tc_flags, NULL);
        }
      } else {
        CB_BLOCKS = cBlocks/BF;
        KB_BLOCKS = kBlocks/BF;

        /* define batch-reduce gemm kernels */
        stride_a = bc * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bc * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->fwd_kernela = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bk, bn, bc, stride_a, stride_b, CB_BLOCKS, &bk, &C, &K, NULL, NULL, &kernel_flags, NULL );
        stride_a = bk * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->fwd_kernelb = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bk, bn, bk, stride_a, stride_b, KB_BLOCKS, &bk, &K, &K, NULL, NULL, &kernel_flags, NULL );
        if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_SPR) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT)) {
          handle->fwd_tileconfig = libxsmm_bsmmdispatch_reducebatch_addr( bk, bn, bk, &bk, &K, &K, NULL, NULL, &tc_flags, NULL );
        }

        BF = handle->bwdupd_block;
        KB_BLOCKS = kBlocks/BF;

        stride_a = bc * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->bwdupd_kernela = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bc, bn, bk, stride_a, stride_b, KB_BLOCKS, &bc, &K, &C, NULL, NULL, &kernel_flags, NULL);
        stride_a = bn * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bn * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->bwdupd_kernelb = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bk, bk, bn, stride_a, stride_b, nBlocks, &bk, &N, &bk, NULL, NULL, &kernel_flags, NULL);
        stride_a = bn * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bn * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->bwdupd_kernelc = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bk, bc, bn, stride_a, stride_b, nBlocks, &bk, &N, &bk, NULL, NULL, &kernel_flags, NULL);
        stride_a = bk * bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        stride_b = bk * libxsmm_dnn_typesize(handle->desc.datatype_in);
        handle->bwdupd_kerneld = libxsmm_bsmmdispatch_reducebatch_strd_unroll( bk, bn, bk, stride_a, stride_b, KB_BLOCKS, &bk, &K, &K, NULL, NULL, &kernel_flags, NULL);
        if ((libxsmm_target_archid == LIBXSMM_X86_AVX512_SPR) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT)) {
          handle->bwdupd_tileconfig = libxsmm_bsmmdispatch_reducebatch_addr( bk, bn, bk, &bk, &K, &K, NULL, NULL, &tc_flags, NULL);
        }
      }
    }

    /* Need to allocate space for scratch libxsmm_dnn_tensor's, let's set all pointers to zero */
    handle->internal_z = 0;
    handle->scratch_wT = 0;
    handle->scratch_rT = 0;
    handle->scratch_xT = 0;
    handle->scratch_hT = 0;
    handle->scratch_deltat = 0;
    handle->scratch_di = 0;
    handle->scratch_df = 0;
    handle->scratch_do = 0;
    handle->scratch_dci = 0;
    handle->scratch_diB = 0;
    handle->scratch_dfB = 0;
    handle->scratch_dpB = 0;
    handle->scratch_dciB = 0;
    /* initialize a high-performant barrier */
    handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
    if (NULL == handle->barrier)
    {
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
      free(handle);
      return NULL;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_rnncell(const libxsmm_dnn_rnncell* handle)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_rnncell*)handle);
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }
  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_rnncell_create_tensor_datalayout(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_tensor_datalayout* layout;
  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;
  if (handle != 0) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    layout = (libxsmm_dnn_tensor_datalayout*)calloc(1, sizeof(libxsmm_dnn_tensor_datalayout));
    if (layout != 0) {
      if ( (type == LIBXSMM_DNN_RNN_REGULAR_INPUT)             || (type == LIBXSMM_DNN_RNN_GRADIENT_INPUT)             ||
           (type == LIBXSMM_DNN_RNN_REGULAR_CS_PREV)           || (type == LIBXSMM_DNN_RNN_GRADIENT_CS_PREV)           ||
           (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV) || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV) ||
           (type == LIBXSMM_DNN_RNN_REGULAR_CS)                || (type == LIBXSMM_DNN_RNN_GRADIENT_CS)                ||
           (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE)      || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE)      ||
           (type == LIBXSMM_DNN_RNN_INTERNAL_I)                || (type == LIBXSMM_DNN_RNN_INTERNAL_F)                 ||
           (type == LIBXSMM_DNN_RNN_INTERNAL_O)                || (type == LIBXSMM_DNN_RNN_INTERNAL_CI)                ||
           (type == LIBXSMM_DNN_RNN_INTERNAL_CO) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXSMM_DNN_ACTIVATION;
        if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32)) || ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)) ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 5;

              if ( (type == LIBXSMM_DNN_RNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_RNN_GRADIENT_INPUT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_T;
                layout->dim_size[0] = (unsigned int)handle->bc;
                layout->dim_size[1] = (unsigned int)handle->bn;
                layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
                layout->dim_size[4] = (unsigned int)handle->desc.max_T;
              } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_CS_PREV)           || (type == LIBXSMM_DNN_RNN_GRADIENT_CS_PREV)           ||
                          (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV) || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV) ||
                          (type == LIBXSMM_DNN_RNN_REGULAR_CS)                || (type == LIBXSMM_DNN_RNN_GRADIENT_CS)                ||
                          (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE)      || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE)      ||
                          (type == LIBXSMM_DNN_RNN_INTERNAL_I)                || (type == LIBXSMM_DNN_RNN_INTERNAL_F)                 ||
                          (type == LIBXSMM_DNN_RNN_INTERNAL_O)                || (type == LIBXSMM_DNN_RNN_INTERNAL_CI)                ||
                          (type == LIBXSMM_DNN_RNN_INTERNAL_CO) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_T;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)handle->bn;
                layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
                layout->dim_size[4] = (unsigned int)handle->desc.max_T;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NC) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32)) || ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)) ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(3*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(3*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 3;

              if ( (type == LIBXSMM_DNN_RNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_RNN_GRADIENT_INPUT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_T;
                layout->dim_size[0] = (unsigned int)handle->desc.C;
                layout->dim_size[1] = (unsigned int)handle->desc.N;
                layout->dim_size[2] = (unsigned int)handle->desc.max_T;
              } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_CS_PREV)           || (type == LIBXSMM_DNN_RNN_GRADIENT_CS_PREV)           ||
                          (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV) || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV) ||
                          (type == LIBXSMM_DNN_RNN_REGULAR_CS)                || (type == LIBXSMM_DNN_RNN_GRADIENT_CS)                ||
                          (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE)      || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE)      ||
                          (type == LIBXSMM_DNN_RNN_INTERNAL_I)                || (type == LIBXSMM_DNN_RNN_INTERNAL_F)                 ||
                          (type == LIBXSMM_DNN_RNN_INTERNAL_O)                || (type == LIBXSMM_DNN_RNN_INTERNAL_CI)                ||
                          (type == LIBXSMM_DNN_RNN_INTERNAL_CO) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_T;
                layout->dim_size[0] = (unsigned int)handle->desc.K;
                layout->dim_size[1] = (unsigned int)handle->desc.N;
                layout->dim_size[2] = (unsigned int)handle->desc.max_T;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT)       || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ||
                  (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
        layout->format = handle->desc.filter_format;
        layout->tensor_type = LIBXSMM_DNN_FILTER;
        if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED) > 0) {
          if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = handle->desc.datatype_in;
            if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM || handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 5;

                if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_X;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bc;
                  layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                    layout->dim_size[4] = 4;
                  } else {
                    layout->dim_size[4] = 3;
                  }
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_X;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                    layout->dim_size[4] = 4;
                  } else {
                    layout->dim_size[4] = 3;
                  }
                } else {
                  free(layout->dim_type);
                  free(layout->dim_size);
                  free(layout);
                  layout = 0; /* make sure a NULL is returned */
                  *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
                }
              } else {
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 4;

                if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bc;
                  layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                } else {
                  free(layout->dim_type);
                  free(layout->dim_size);
                  free(layout);
                  layout = 0; /* make sure a NULL is returned */
                  *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
                }
              } else {
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = handle->desc.datatype_in;
            if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM || handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 6;

                if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_X;
                  layout->dim_size[0] = (unsigned int)handle->lpb;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->bc / handle->lpb);
                  layout->dim_size[3] = (unsigned int)(handle->desc.C / handle->bc);
                  layout->dim_size[4] = (unsigned int)(handle->desc.K / handle->bk);
                  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                    layout->dim_size[5] = 4;
                  } else {
                    layout->dim_size[5] = 3;
                  }
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_X;
                  layout->dim_size[0] = (unsigned int)handle->lpb;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->bk / handle->lpb);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[4] = (unsigned int)(handle->desc.K / handle->bk);
                  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                    layout->dim_size[5] = 4;
                  } else {
                    layout->dim_size[5] = 3;
                  }
                } else {
                  free(layout->dim_type);
                  free(layout->dim_size);
                  free(layout);
                  layout = 0; /* make sure a NULL is returned */
                  *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
                }
              } else {
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 5;

                if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_size[0] = (unsigned int)handle->lpb;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->bc / handle->lpb);
                  layout->dim_size[3] = (unsigned int)(handle->desc.C / handle->bc);
                  layout->dim_size[4] = (unsigned int)(handle->desc.K / handle->bk);
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_size[0] = (unsigned int)handle->lpb;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->bk / handle->lpb);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[4] = (unsigned int)(handle->desc.K / handle->bk);
                } else {
                  free(layout->dim_type);
                  free(layout->dim_size);
                  free(layout);
                  layout = 0; /* make sure a NULL is returned */
                  *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
                }
              } else {
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }

          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_CK) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32)) || ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)) ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;

              if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                  layout->dim_size[0] = (unsigned int)(handle->desc.K * 4);
                  layout->dim_size[1] = (unsigned int)handle->desc.C;
                } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
                  layout->dim_size[0] = (unsigned int)(handle->desc.K * 3);
                  layout->dim_size[1] = (unsigned int)handle->desc.C;
                } else {
                  layout->dim_size[0] = (unsigned int)handle->desc.K;
                  layout->dim_size[1] = (unsigned int)handle->desc.C;
                }
              } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                  layout->dim_size[0] = (unsigned int)(handle->desc.K * 4);
                  layout->dim_size[1] = (unsigned int)handle->desc.K;
                } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
                  layout->dim_size[0] = (unsigned int)(handle->desc.K * 3);
                  layout->dim_size[1] = (unsigned int)handle->desc.K;
                } else {
                  layout->dim_size[0] = (unsigned int)handle->desc.K;
                  layout->dim_size[1] = (unsigned int)handle->desc.K;
                }
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS) || (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS) ) {
        layout->format = handle->desc.filter_format;
        layout->tensor_type = LIBXSMM_DNN_FILTER;
        if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED) > 0) {
          if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = handle->desc.datatype_in;
            if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM || handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 5;

                if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_X;
                  layout->dim_size[0] = (unsigned int)handle->bc;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.C / handle->bc);
                  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                    layout->dim_size[4] = 4;
                  } else {
                    layout->dim_size[4] = 3;
                  }
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_X;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                    layout->dim_size[4] = 4;
                  } else {
                    layout->dim_size[4] = 3;
                  }
                } else {
                  free(layout->dim_type);
                  free(layout->dim_size);
                  free(layout);
                  layout = 0; /* make sure a NULL is returned */
                  *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
                }
              } else {
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 4;

                if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_size[0] = (unsigned int)handle->bc;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.C / handle->bc);
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                } else {
                  free(layout->dim_type);
                  free(layout->dim_size);
                  free(layout);
                  layout = 0; /* make sure a NULL is returned */
                  *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
                }
              } else {
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = handle->desc.datatype_in;
            if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM || handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 6;

                if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_X;
                  layout->dim_size[0] = (unsigned int)handle->lpb;
                  layout->dim_size[1] = (unsigned int)handle->bc;
                  layout->dim_size[2] = (unsigned int)(handle->bk / handle->lpb);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[4] = (unsigned int)(handle->desc.C / handle->bc);
                  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                    layout->dim_size[5] = 4;
                  } else {
                    layout->dim_size[5] = 3;
                  }
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_X;
                  layout->dim_size[0] = (unsigned int)handle->lpb;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->bk / handle->lpb);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[4] = (unsigned int)(handle->desc.K / handle->bk);
                  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                    layout->dim_size[5] = 4;
                  } else {
                    layout->dim_size[5] = 3;
                  }
                } else {
                  free(layout->dim_type);
                  free(layout->dim_size);
                  free(layout);
                  layout = 0; /* make sure a NULL is returned */
                  *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
                }
              } else {
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 5;

                if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                  layout->dim_size[0] = (unsigned int)handle->lpb;
                  layout->dim_size[1] = (unsigned int)handle->bc;
                  layout->dim_size[2] = (unsigned int)(handle->bk / handle->lpb);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[4] = (unsigned int)(handle->desc.C / handle->bc);
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                  layout->dim_size[0] = (unsigned int)handle->lpb;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->bk / handle->lpb);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[4] = (unsigned int)(handle->desc.K / handle->bk);
                } else {
                  free(layout->dim_type);
                  free(layout->dim_size);
                  free(layout);
                  layout = 0; /* make sure a NULL is returned */
                  *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
                }
              } else {
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_CK) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32)) || ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)) ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;

              if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                  layout->dim_size[0] = (unsigned int)handle->desc.C;
                  layout->dim_size[1] = (unsigned int)(handle->desc.K * 4);
                } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
                  layout->dim_size[0] = (unsigned int)handle->desc.C;
                  layout->dim_size[1] = (unsigned int)(handle->desc.K * 3);
                } else {
                  layout->dim_size[0] = (unsigned int)handle->desc.C;
                  layout->dim_size[1] = (unsigned int)handle->desc.K;
                }
              } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                  layout->dim_size[0] = (unsigned int)handle->desc.K;
                  layout->dim_size[1] = (unsigned int)(handle->desc.K * 4);
                } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
                  layout->dim_size[0] = (unsigned int)handle->desc.K;
                  layout->dim_size[1] = (unsigned int)(handle->desc.K * 3);
                } else {
                  layout->dim_size[0] = (unsigned int)handle->desc.K;
                  layout->dim_size[1] = (unsigned int)handle->desc.K;
                }
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_BIAS) || (type == LIBXSMM_DNN_RNN_GRADIENT_BIAS) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXSMM_DNN_CHANNEL_SCALAR;


        if ( ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NC) > 0) || ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) > 0) ) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32)) || ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)) ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(1*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(1*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 1;

              if ( (type == LIBXSMM_DNN_RNN_REGULAR_BIAS) || (type == LIBXSMM_DNN_RNN_GRADIENT_BIAS) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
                  layout->dim_size[0] = (unsigned int)(handle->desc.K * 4);
                } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
                  layout->dim_size[0] = (unsigned int)(handle->desc.K * 3);
                } else {
                  layout->dim_size[0] = (unsigned int)handle->desc.K;
                }
              } else { /* coverity[dead_error_begin] */
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else {
        free(layout);
        layout = 0; /* make sure a NULL is returned */
        *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
      }
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }
  return layout;
}


LIBXSMM_API size_t libxsmm_dnn_rnncell_get_scratch_size(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    const size_t typesize_in = libxsmm_dnn_typesize(handle->desc.datatype_in);
    const size_t dwdr_typesize = (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ? sizeof(float) : typesize_in;

    switch (handle->desc.cell_type) {
      case LIBXSMM_DNN_RNNCELL_RNN_RELU:
      case LIBXSMM_DNN_RNNCELL_RNN_SIGMOID:
      case LIBXSMM_DNN_RNNCELL_RNN_TANH: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            size += 0;
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            size += (size_t)handle->desc.C * (size_t)handle->desc.K * typesize_in  + 64; /* wT */
            size += (size_t)handle->desc.K * (size_t)handle->desc.K * typesize_in  + 64; /* rT */
            size += (size_t)handle->desc.C * (size_t)handle->desc.N * typesize_in  + 64; /* xT */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* hT */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) * (size_t)handle->desc.max_T + 64; /* deltat */

          } break;
          default: {
            *status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case  LIBXSMM_DNN_RNNCELL_LSTM: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            size += (size_t)handle->desc.C * (size_t)handle->desc.K * typesize_in * 4 + 4 * 64; /* w */
            size += (size_t)handle->desc.K * (size_t)handle->desc.K * typesize_in * 4 + 4 * 64; /* r */
            /*  The scratches below are needed only for BF16 code for the intermediate results  */
            if (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) {
              size += (size_t)7 *((size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64); /* intermediate scratches */
              size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) + 64;                                           /* intermediate scratches */
            }
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            size += (size_t)handle->desc.C * (size_t)handle->desc.K * dwdr_typesize * 4 + 4 * 64; /* w */
            size += (size_t)handle->desc.K * (size_t)handle->desc.K * dwdr_typesize * 4 + 4 * 64; /* r */
            size += (size_t)handle->desc.C * (size_t)handle->desc.K * typesize_in * 4 + 4 * 64; /* wT */
            size += (size_t)handle->desc.K * (size_t)handle->desc.K * typesize_in * 4 + 4 * 64; /* rT */
            size += (size_t)handle->desc.C * (size_t)handle->desc.N * typesize_in  + 64; /* xT */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* hT */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * dwdr_typesize + 64; /* deltat */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* di */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* df */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* do */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* dci */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* diB */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* dfB */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* dpB */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* dciB */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* t1 */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* t2 */
            /*  The scratches below are needed only for BF16 code for the intermediate results  */
            if (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) {
              size += (size_t)4 *((size_t)handle->desc.K * sizeof(float) + 64); /* intermediate db scratch */
              size += (size_t)handle->desc.C * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64; /* intermediate dx scratches */
              size += (size_t)7 *((size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64); /* intermediate scratches */
              size += (size_t)2 *((size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) + 64); /* intermediate scratches */
            }
          } break;
          default: {
            *status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case  LIBXSMM_DNN_RNNCELL_GRU: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            size += (size_t)handle->desc.C * (size_t)handle->desc.K * typesize_in * 3 + 3 * 64; /* w */
            size += (size_t)handle->desc.K * (size_t)handle->desc.K * typesize_in * 3 + 3 * 64; /* r */
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            size += (size_t)handle->desc.C * (size_t)handle->desc.K * dwdr_typesize * 3 + 3 * 64; /* w */
            size += (size_t)handle->desc.K * (size_t)handle->desc.K * dwdr_typesize * 3 + 3 * 64; /* r */
            size += (size_t)handle->desc.C * (size_t)handle->desc.K * typesize_in * 3 + 3 * 64; /* wT */
            size += (size_t)handle->desc.K * (size_t)handle->desc.K * typesize_in * 3 + 3 * 64; /* rT */
            size += (size_t)handle->desc.C * (size_t)handle->desc.N * typesize_in  + 64; /* xT */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* hT */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * dwdr_typesize + 64; /* deltat */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* di */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* dc */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* df */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* do */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* diB */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* dcB */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* dfB */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* oT */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* t1 */
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64; /* t2 */
          } break;
          default: {
            *status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      default: {
        *status = LIBXSMM_DNN_ERR_INVALID_RNN_TYPE;
      }
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return size;
}


LIBXSMM_API void* libxsmm_dnn_rnncell_get_scratch_ptr(const libxsmm_dnn_rnncell* handle, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    return handle->scratch_base;
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return NULL;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_scratch(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, const void* scratch)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (NULL != handle) {
    const size_t typesize_in = libxsmm_dnn_typesize(handle->desc.datatype_in);
    const size_t dwdr_typesize = (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ? sizeof(float) : typesize_in;
    uintptr_t address = (uintptr_t)scratch;
    size_t offset = 0;

    switch (handle->desc.cell_type) {
      case LIBXSMM_DNN_RNNCELL_RNN_RELU:
      case LIBXSMM_DNN_RNNCELL_RNN_SIGMOID:
      case LIBXSMM_DNN_RNNCELL_RNN_TANH: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            /* forward only has no scratch need */
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            if (scratch == 0) {
              status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
              return status;
            }
            handle->scratch_base = (void*)address;
            /* wT */
            if (address % 64 == 0) {
              handle->scratch_wT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_wT = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.C * (size_t)handle->desc.K * typesize_in) + 64;
            /* rT */
            if (address % 64 == 0) {
              handle->scratch_rT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_rT = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.K * (size_t)handle->desc.K * typesize_in) + 64;
            /* xT */
            if (address % 64 == 0) {
              handle->scratch_xT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_xT = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.C * (size_t)handle->desc.N * typesize_in) + 64;
            /* hT */
            if (address % 64 == 0) {
              handle->scratch_hT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_hT = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out)) + 64;
            /* deltat */
            if (address % 64 == 0) {
              handle->scratch_deltat = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_deltat = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) * (size_t)handle->desc.max_T) + 64;
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_RNNCELL_LSTM: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            if (scratch == 0) {
              status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
              return status;
            }
            handle->scratch_base = (void*)address;
            /* w scratch */
            if (address % 64 == 0) {
              handle->scratch_w = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_w = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.C * (size_t)handle->desc.K * typesize_in) * 4 + 64;
            /* r scratch */
            if (address % 64 == 0) {
              handle->scratch_r = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_r = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.K * (size_t)handle->desc.K * typesize_in) * 4 + 64;
            /*  The scratches below are needed only for BF16 code for the intermediate results  */
            if (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) {
              /* cst scratch */
              if (address % 64 == 0) {
                handle->cst_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->cst_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* ht scratch */
              if (address % 64 == 0) {
                handle->ht_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->ht_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* it scratch */
              if (address % 64 == 0) {
                handle->it_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->it_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* ft scratch */
              if (address % 64 == 0) {
                handle->ft_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->ft_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* ot scratch */
              if (address % 64 == 0) {
                handle->ot_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->ot_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* cit scratch */
              if (address % 64 == 0) {
                handle->cit_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->cit_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* cot scratch */
              if (address % 64 == 0) {
                handle->cot_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->cot_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* csp scratch */
              if (address % 64 == 0) {
                handle->csp_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->csp_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) + 64;
            }
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            if (scratch == 0) {
              status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
              return status;
            }
            handle->scratch_base = (void*)address;
            /* w scratch */
            if (address % 64 == 0) {
              handle->scratch_w = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_w = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.C * (size_t)handle->desc.K * dwdr_typesize) * 4 + 64;
            /* r scratch */
            if (address % 64 == 0) {
              handle->scratch_r = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_r = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.K * (size_t)handle->desc.K * dwdr_typesize) * 4 + 64;
            /* wT */
            if (address % 64 == 0) {
              handle->scratch_wT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_wT = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.C * (size_t)handle->desc.K * typesize_in) * 4 + 64;
            /* rT */
            if (address % 64 == 0) {
              handle->scratch_rT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_rT = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.K * (size_t)handle->desc.K * typesize_in) * 4 + 64;
            /* xT */
            if (address % 64 == 0) {
              handle->scratch_xT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_xT = (void*)(address+offset);
            }
            address += (size_t)handle->desc.C * (size_t)handle->desc.N * typesize_in + 64;
            /* hT */
            if (address % 64 == 0) {
              handle->scratch_hT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_hT = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* deltat */
            if (address % 64 == 0) {
              handle->scratch_deltat = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_deltat = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * dwdr_typesize + 64;
            /* di */
            if (address % 64 == 0) {
              handle->scratch_di = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_di = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* df */
            if (address % 64 == 0) {
              handle->scratch_df = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_df = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* do */
            if (address % 64 == 0) {
              handle->scratch_do = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_do = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* dci */
            if (address % 64 == 0) {
              handle->scratch_dci = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_dci = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* diB */
            if (address % 64 == 0) {
              handle->scratch_diB = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_diB = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* dfB */
            if (address % 64 == 0) {
              handle->scratch_dfB = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_dfB = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* dpB */
            if (address % 64 == 0) {
              handle->scratch_dpB = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_dpB = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* dciB */
            if (address % 64 == 0) {
              handle->scratch_dciB = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_dciB = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* t1 */
            if (address % 64 == 0) {
              handle->scratch_t1 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_t1 = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* t2 */
            if (address % 64 == 0) {
              handle->scratch_t2 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_t2 = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /*  The scratches below are needed only for BF16 code for the intermediate results  */
            if (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) {
              /* dx scratch */
              if (address % 64 == 0) {
                handle->scratch_dx = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->scratch_dx = (void*)(address+offset);
              }
              address += (size_t)handle->desc.C * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* dhp scratch */
              if (address % 64 == 0) {
                handle->scratch_dhp = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->scratch_dhp = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) + 64;
              /* db scratch */
              if (address % 64 == 0) {
                handle->scratch_db = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->scratch_db = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * 4 * sizeof(float) + 64;
              /* cst scratch */
              if (address % 64 == 0) {
                handle->cst_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->cst_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* ht scratch */
              if (address % 64 == 0) {
                handle->ht_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->ht_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* it scratch */
              if (address % 64 == 0) {
                handle->it_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->it_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* ft scratch */
              if (address % 64 == 0) {
                handle->ft_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->ft_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* ot scratch */
              if (address % 64 == 0) {
                handle->ot_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->ot_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* cit scratch */
              if (address % 64 == 0) {
                handle->cit_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->cit_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* cot scratch */
              if (address % 64 == 0) {
                handle->cot_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->cot_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) * (size_t)handle->desc.max_T + 64;
              /* csp scratch */
              if (address % 64 == 0) {
                handle->csp_scratch = (void*)address;
              } else {
                offset = (64 - address % 64);
                handle->csp_scratch = (void*)(address+offset);
              }
              address += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof(float) + 64;
            }
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_RNNCELL_GRU: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            if (scratch == 0) {
              status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
              return status;
            }
            handle->scratch_base = (void*)address;
            /* w scratch */
            if (address % 64 == 0) {
              handle->scratch_w = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_w = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.C * (size_t)handle->desc.K * typesize_in) * 3 + 64;
            /* r scratch */
            if (address % 64 == 0) {
              handle->scratch_r = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_r = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.K * (size_t)handle->desc.K * typesize_in) * 3 + 64;
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            if (scratch == 0) {
              status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
              return status;
            }
            handle->scratch_base = (void*)address;
            /* w scratch */
            if (address % 64 == 0) {
              handle->scratch_w = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_w = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.C * (size_t)handle->desc.K * dwdr_typesize) * 3 + 64;
            /* r scratch */
            if (address % 64 == 0) {
              handle->scratch_r = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_r = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.K * (size_t)handle->desc.K * dwdr_typesize) * 3 + 64;
            /* wT */
            if (address % 64 == 0) {
              handle->scratch_wT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_wT = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.C * (size_t)handle->desc.K * typesize_in) * 3 + 64;
            /* rT */
            if (address % 64 == 0) {
              handle->scratch_rT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_rT = (void*)(address+offset);
            }
            address += ((size_t)handle->desc.K * (size_t)handle->desc.K * typesize_in) * 3 + 64;
            /* xT */
            if (address % 64 == 0) {
              handle->scratch_xT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_xT = (void*)(address+offset);
            }
            address += (size_t)handle->desc.C * (size_t)handle->desc.N * typesize_in + 64;
            /* hT */
            if (address % 64 == 0) {
              handle->scratch_hT = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_hT = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* deltat */
            if (address % 64 == 0) {
              handle->scratch_deltat = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_deltat = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * dwdr_typesize + 64;
            /* di */
            if (address % 64 == 0) {
              handle->scratch_di = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_di = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* dc */
            if (address % 64 == 0) {
              handle->scratch_dci = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_dci = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* df */
            if (address % 64 == 0) {
              handle->scratch_df = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_df = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* do */
            if (address % 64 == 0) {
              handle->scratch_do = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_do = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* diB */
            if (address % 64 == 0) {
              handle->scratch_diB = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_diB = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* dcB */
            if (address % 64 == 0) {
              handle->scratch_dciB = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_dciB = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* dfB */
            if (address % 64 == 0) {
              handle->scratch_dfB = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_dfB = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* doB (repurposed for oT) */
            if (address % 64 == 0) {
              handle->scratch_dpB = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_dpB = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* t1 */
            if (address % 64 == 0) {
              handle->scratch_t1 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_t1 = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
            /* t2 */
            if (address % 64 == 0) {
              handle->scratch_t2 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch_t2 = (void*)(address+offset);
            }
            address += (size_t)handle->desc.K * (size_t)handle->desc.N * libxsmm_dnn_typesize(handle->desc.datatype_out) + 64;
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_RNN_TYPE;
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_scratch(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (handle->desc.cell_type) {
      case LIBXSMM_DNN_RNNCELL_RNN_RELU:
      case LIBXSMM_DNN_RNNCELL_RNN_SIGMOID:
      case LIBXSMM_DNN_RNNCELL_RNN_TANH: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            /* forward only has no scratch need */
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            handle->scratch_wT = 0;
            handle->scratch_rT = 0;
            handle->scratch_xT = 0;
            handle->scratch_hT = 0;
            handle->scratch_deltat = 0;
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_RNNCELL_LSTM: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            handle->scratch_w  = 0;
            handle->scratch_r  = 0;
            handle->csp_scratch  = 0;
            handle->cst_scratch  = 0;
            handle->ht_scratch  = 0;
            handle->it_scratch  = 0;
            handle->ft_scratch  = 0;
            handle->ot_scratch  = 0;
            handle->cit_scratch  = 0;
            handle->cot_scratch  = 0;
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            handle->scratch_w = 0;
            handle->scratch_r = 0;
            handle->scratch_wT = 0;
            handle->scratch_rT = 0;
            handle->scratch_xT = 0;
            handle->scratch_hT = 0;
            handle->scratch_deltat = 0;
            handle->scratch_di = 0;
            handle->scratch_df = 0;
            handle->scratch_do = 0;
            handle->scratch_dci = 0;
            handle->scratch_diB = 0;
            handle->scratch_dfB = 0;
            handle->scratch_dpB = 0;
            handle->scratch_dciB = 0;
            handle->scratch_t1 = 0;
            handle->scratch_t2 = 0;
            handle->csp_scratch = 0;
            handle->cst_scratch = 0;
            handle->ht_scratch = 0;
            handle->it_scratch = 0;
            handle->ft_scratch = 0;
            handle->ot_scratch = 0;
            handle->cit_scratch = 0;
            handle->cot_scratch = 0;
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_RNNCELL_GRU: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            handle->scratch_w   = 0;
            handle->scratch_r   = 0;
            handle->ht_scratch  = 0;
            handle->it_scratch  = 0;
            handle->cit_scratch = 0;
            handle->ft_scratch  = 0;
            handle->ot_scratch  = 0;
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            handle->scratch_w  = 0;
            handle->scratch_r  = 0;
            handle->scratch_wT = 0;
            handle->scratch_rT = 0;
            handle->scratch_xT = 0;
            handle->scratch_hT = 0;
            handle->scratch_deltat = 0;
            handle->scratch_di = 0;
            handle->scratch_dci = 0;
            handle->scratch_df  = 0;
            handle->scratch_do  = 0;
            handle->scratch_diB = 0;
            handle->scratch_dciB = 0;
            handle->scratch_dfB = 0;
            handle->scratch_dpB = 0;
            handle->scratch_t1  = 0;
            handle->scratch_t2  = 0;
            handle->ht_scratch  = 0;
            handle->it_scratch  = 0;
            handle->ft_scratch  = 0;
            handle->ot_scratch  = 0;
            handle->cit_scratch = 0;
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_RNN_TYPE;
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API size_t libxsmm_dnn_rnncell_get_internalstate_size(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    const size_t sizeof_datatype = sizeof(float);

    switch (handle->desc.cell_type) {
      case LIBXSMM_DNN_RNNCELL_RNN_RELU:
      case LIBXSMM_DNN_RNNCELL_RNN_SIGMOID:
      case LIBXSMM_DNN_RNNCELL_RNN_TANH: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.max_T + 64; /* zt */
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.max_T + 64; /* zt */
          } break;
          default: {
            *status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_RNNCELL_LSTM: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            /* with i, f, o, ci, co, cs exposed as i/o, there is currently no need for internal state */
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            /* with i, f, o, ci, co, cs exposed as i/o, there is currently no need for internal state */
          } break;
          default: {
            *status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_RNNCELL_GRU: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            /* with i, f, c, o exposed as i/o, there is currently no need for internal state */
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            /* with i, f, c, o exposed as i/o, there is currently no need for internal state */
          } break;
          default: {
            *status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      default: {
        *status = LIBXSMM_DNN_ERR_INVALID_RNN_TYPE;
      }
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return size;
}


LIBXSMM_API void* libxsmm_dnn_rnncell_get_internalstate_ptr(const libxsmm_dnn_rnncell* handle, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    return handle->internal_z;
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return NULL;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_internalstate(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, const void* internalstate)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)internalstate;
  size_t offset = 0;

  if (0 != handle) {
    switch (handle->desc.cell_type) {
      case LIBXSMM_DNN_RNNCELL_RNN_RELU:
      case LIBXSMM_DNN_RNNCELL_RNN_SIGMOID:
      case LIBXSMM_DNN_RNNCELL_RNN_TANH: {
        if (internalstate == 0) {
          status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
          return status;
        }
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            if (address % 64 == 0) {
              handle->internal_z = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->internal_z = (void*)(address+offset);
            }
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            if (address % 64 == 0) {
              handle->internal_z = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->internal_z = (void*)(address+offset);
            }
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_RNNCELL_LSTM: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_RNNCELL_GRU: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_RNN_TYPE;
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_internalstate(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (handle->desc.cell_type) {
      case LIBXSMM_DNN_RNNCELL_RNN_RELU:
      case LIBXSMM_DNN_RNNCELL_RNN_SIGMOID:
      case LIBXSMM_DNN_RNNCELL_RNN_TANH: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            handle->internal_z = 0;
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
            handle->internal_z = 0;
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_RNNCELL_LSTM: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_RNNCELL_GRU: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD:
          case LIBXSMM_DNN_COMPUTE_KIND_UPD:
          case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD:
          case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_RNN_TYPE;
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_allocate_forget_bias(libxsmm_dnn_rnncell* handle, const float forget_bias)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0) {
    handle->forget_bias = forget_bias;
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_RNN_REGULAR_INPUT)             && (type != LIBXSMM_DNN_RNN_GRADIENT_INPUT)             &&
       (type != LIBXSMM_DNN_RNN_REGULAR_CS_PREV)           && (type != LIBXSMM_DNN_RNN_GRADIENT_CS_PREV)           &&
       (type != LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV) && (type != LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV) &&
       (type != LIBXSMM_DNN_RNN_REGULAR_WEIGHT)            && (type != LIBXSMM_DNN_RNN_GRADIENT_WEIGHT)            &&
       (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT)      && (type != LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT)      &&
       (type != LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS)      && (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS) &&
       (type != LIBXSMM_DNN_RNN_REGULAR_BIAS)              && (type != LIBXSMM_DNN_RNN_GRADIENT_BIAS)              &&
       (type != LIBXSMM_DNN_RNN_REGULAR_CS)                && (type != LIBXSMM_DNN_RNN_GRADIENT_CS)                &&
       (type != LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE)      && (type != LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE)      &&
       (type != LIBXSMM_DNN_RNN_INTERNAL_I)                && (type != LIBXSMM_DNN_RNN_INTERNAL_F)                 &&
       (type != LIBXSMM_DNN_RNN_INTERNAL_O)                && (type != LIBXSMM_DNN_RNN_INTERNAL_CI)                &&
       (type != LIBXSMM_DNN_RNN_INTERNAL_CO) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_rnncell_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_RNN_REGULAR_INPUT ) {
        handle->xt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_INPUT ) {
        handle->dxt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_CS_PREV ) {
        handle->csp = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_CS_PREV ) {
        handle->dcsp = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) {
        handle->hp = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV ) {
        handle->dhp = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) {
        handle->w = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS ) {
        handle->wt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) {
        handle->dw = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) {
        handle->r = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS ) {
        handle->rt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) {
        handle->dr = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_BIAS ) {
        handle->b = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_BIAS ) {
        handle->db = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_CS ) {
        handle->cst = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_CS ) {
        handle->dcs = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) {
        handle->ht = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) {
        handle->dht = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_I ) {
        handle->it = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_F ) {
        handle->ft = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_O ) {
        handle->ot = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_CI ) {
        handle->cit = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_CO ) {
        handle->cot = (libxsmm_dnn_tensor*)tensor;
      } else {
        /* cannot happen */
      }
    } else {
      status = LIBXSMM_DNN_ERR_MISMATCH_TENSOR;
    }

    libxsmm_dnn_destroy_tensor_datalayout( handle_layout );
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_rnncell_get_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_tensor* tensor = 0;
  LIBXSMM_UNUSED(status/*TODO*/);

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_RNN_REGULAR_INPUT)             && (type != LIBXSMM_DNN_RNN_GRADIENT_INPUT)             &&
       (type != LIBXSMM_DNN_RNN_REGULAR_CS_PREV)           && (type != LIBXSMM_DNN_RNN_GRADIENT_CS_PREV)           &&
       (type != LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV) && (type != LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV) &&
       (type != LIBXSMM_DNN_RNN_REGULAR_WEIGHT)            && (type != LIBXSMM_DNN_RNN_GRADIENT_WEIGHT)            &&
       (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT)      && (type != LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT)      &&
       (type != LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS)      && (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS) &&
       (type != LIBXSMM_DNN_RNN_REGULAR_BIAS)              && (type != LIBXSMM_DNN_RNN_GRADIENT_BIAS)              &&
       (type != LIBXSMM_DNN_RNN_REGULAR_CS)                && (type != LIBXSMM_DNN_RNN_GRADIENT_CS)                &&
       (type != LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE)      && (type != LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE)      &&
       (type != LIBXSMM_DNN_RNN_INTERNAL_I)                && (type != LIBXSMM_DNN_RNN_INTERNAL_F)                 &&
       (type != LIBXSMM_DNN_RNN_INTERNAL_O)                && (type != LIBXSMM_DNN_RNN_INTERNAL_CI)                &&
       (type != LIBXSMM_DNN_RNN_INTERNAL_CO) ) {
    return tensor;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_RNN_REGULAR_INPUT ) {
      tensor = handle->xt;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_INPUT ) {
      tensor = handle->dxt;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_CS_PREV ) {
      tensor = handle->csp;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_CS_PREV ) {
      tensor = handle->dcsp;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) {
      tensor = handle->hp;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV ) {
      tensor = handle->dhp;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) {
      tensor = handle->w;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS ) {
      tensor = handle->wt;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) {
      tensor = handle->dw;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) {
      tensor = handle->r;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS ) {
      tensor = handle->rt;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) {
      tensor = handle->dr;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_BIAS ) {
      tensor = handle->b;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_BIAS ) {
      tensor = handle->db;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_CS ) {
      tensor = handle->cst;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_CS ) {
      tensor = handle->dcs;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) {
      tensor = handle->ht;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) {
      tensor = handle->dht;
    } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_I ) {
      tensor = handle->it;
    } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_F ) {
      tensor = handle->ft;
    } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_O ) {
      tensor = handle->ot;
    } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_CI ) {
      tensor = handle->cit;
    } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_CO ) {
      tensor = handle->cot;
    } else {
      /* cannot happen */
    }
  }

  return tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_RNN_REGULAR_INPUT)             && (type != LIBXSMM_DNN_RNN_GRADIENT_INPUT)             &&
       (type != LIBXSMM_DNN_RNN_REGULAR_CS_PREV)           && (type != LIBXSMM_DNN_RNN_GRADIENT_CS_PREV)           &&
       (type != LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV) && (type != LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV) &&
       (type != LIBXSMM_DNN_RNN_REGULAR_WEIGHT)            && (type != LIBXSMM_DNN_RNN_GRADIENT_WEIGHT)            &&
       (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT)      && (type != LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT)      &&
       (type != LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS)      && (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS) &&
       (type != LIBXSMM_DNN_RNN_REGULAR_BIAS)              && (type != LIBXSMM_DNN_RNN_GRADIENT_BIAS)              &&
       (type != LIBXSMM_DNN_RNN_REGULAR_CS)                && (type != LIBXSMM_DNN_RNN_GRADIENT_CS)                &&
       (type != LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE)      && (type != LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE)      &&
       (type != LIBXSMM_DNN_RNN_INTERNAL_I)                && (type != LIBXSMM_DNN_RNN_INTERNAL_F)                 &&
       (type != LIBXSMM_DNN_RNN_INTERNAL_O)                && (type != LIBXSMM_DNN_RNN_INTERNAL_CI)                &&
       (type != LIBXSMM_DNN_RNN_INTERNAL_CO) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_RNN_REGULAR_INPUT ) {
      handle->xt = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_INPUT ) {
      handle->dxt = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_CS_PREV ) {
      handle->csp = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_CS_PREV ) {
      handle->dcsp = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) {
      handle->hp = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV ) {
      handle->dhp = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) {
      handle->w = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS ) {
      handle->wt = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) {
      handle->dw = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) {
      handle->r = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS ) {
      handle->rt = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) {
      handle->dr = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_BIAS ) {
      handle->b = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_BIAS ) {
      handle->db = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_CS ) {
      handle->cst = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_CS ) {
      handle->dcs = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) {
      handle->ht = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) {
      handle->dht = 0;
    } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_I ) {
      handle->it = 0;
    } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_F ) {
      handle->ft = 0;
    } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_O ) {
      handle->ot = 0;
    } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_CI ) {
      handle->cit = 0;
    } else if ( type == LIBXSMM_DNN_RNN_INTERNAL_CO ) {
      handle->cot = 0;
    } else {
      /* cannot happen */
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_set_sequence_length( libxsmm_dnn_rnncell* handle, const libxsmm_blasint T ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    if ( handle->desc.max_T < T ) {
      status = LIBXSMM_DNN_ERR_RNN_INVALID_SEQ_LEN;
    } else {
      handle->T = T;
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_blasint libxsmm_dnn_rnncell_get_sequence_length( libxsmm_dnn_rnncell* handle, libxsmm_dnn_err_t* status ) {
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    return handle->T;
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return 0;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_execute_st(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
        if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NC) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_CK) ) {
          status = libxsmm_dnn_rnncell_st_fwd_nc_ck( handle, start_thread, tid );
        } else if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NC) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED)  ) {
          status = libxsmm_dnn_rnncell_st_fwd_nc_kcck( handle, start_thread, tid );
        } else if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED)  ) {
          status = libxsmm_dnn_rnncell_st_fwd_ncnc_kcck( handle, start_thread, tid );
        } else {
          status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_BWDUPD: {
        if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NC) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_CK) ) {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_ck( handle, kind, start_thread, tid );
        } else if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NC) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED)  ) {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_kcck( handle, kind, start_thread, tid );
        } else if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED)  ) {
          status = libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck( handle, kind, start_thread, tid );
        } else {
          status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_KIND;
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}

