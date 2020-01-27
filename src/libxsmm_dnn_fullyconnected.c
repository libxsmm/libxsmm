/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_fullyconnected_weight_update.h"
#include "libxsmm_dnn_fullyconnected_backward.h"
#include "libxsmm_dnn_fullyconnected_forward.h"
#include "libxsmm_main.h"
#define STRIDE_BRGEMM

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API libxsmm_dnn_fullyconnected* libxsmm_dnn_create_fullyconnected(libxsmm_dnn_fullyconnected_desc fullyconnected_desc, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_fullyconnected* handle = 0;

  if ( ((fullyconnected_desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (fullyconnected_desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)) ||
       ((fullyconnected_desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32)  && (fullyconnected_desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32))  ||
       ((fullyconnected_desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (fullyconnected_desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32))     ) {
    handle = (libxsmm_dnn_fullyconnected*)malloc(sizeof(libxsmm_dnn_fullyconnected));

    if (0 != handle) {
      *status = LIBXSMM_DNN_SUCCESS;
      /* zero entire content; not only safer but also sets data and code pointers to NULL */
      memset(handle, 0, sizeof(*handle));
      /* let's make the description persistent */
      handle->desc = fullyconnected_desc;
      /* @TODO perhaps we need a better switch here */
      if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED) ) {
        handle->bk = handle->desc.bk;
        handle->bn = handle->desc.bn;
        handle->bc = handle->desc.bc;

        if ( handle->desc.N % handle->bn != 0 ) {
          handle->bn = handle->desc.N;
          *status = LIBXSMM_DNN_WARN_FC_SUBOPTIMAL_N_BLOCKING;
        }
        if ( handle->desc.C % handle->bc != 0 ) {
          handle->bc = handle->desc.C;
          *status = LIBXSMM_DNN_WARN_FC_SUBOPTIMAL_C_BLOCKING;
        }
        if ( handle->desc.K % handle->bk != 0 ) {
          handle->bk = handle->desc.K;
          *status = LIBXSMM_DNN_WARN_FC_SUBOPTIMAL_K_BLOCKING;
        }

        /* TODO: Consider subtasking parameters here for UPD pass based on input sizes... */
        handle->ifm_subtasks = atoi(getenv("IFM_SUBTASKS"));
        handle->ofm_subtasks = atoi(getenv("OFM_SUBTASKS"));
      } else {
        /* we need to compute the memory layout given the */
        if ( (handle->desc.C % 16 == 0) && (handle->desc.K % 16 == 0) ) {
          if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            *status = libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.K,
                                                          &(handle->ifmblock), &(handle->ofmblock), &(handle->fm_lp_block),
                                                          LIBXSMM_DNN_DATATYPE_F32, LIBXSMM_DNN_DATATYPE_F32 );
          } else if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            *status = libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.K,
                                                          &(handle->ifmblock), &(handle->ofmblock), &(handle->fm_lp_block),
                                                          handle->desc.datatype_in, handle->desc.datatype_out );
          } else {
            /* should not happen, not implemented */
          }
        } else if ( (handle->desc.C % 64 == 0) && (handle->desc.K == 1000) ) {
          /* @TODO this a hack for the last FC layer */
          handle->ifmblock = 64;
          handle->fm_lp_block = 1;
          handle->ofmblock = 10;
        } else if ( (handle->desc.C % 16 == 0) && (handle->desc.K == 1000) ) {
          /* @TODO this a hack for the last FC layer */
          handle->ifmblock = 16;
          handle->fm_lp_block = 1;
          handle->ofmblock = 10;
        } else {
          *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
          free( handle );
          return 0;
        }
        /* compute the outer blocks */
        handle->blocksifm = handle->desc.C / handle->ifmblock;
        handle->blocksofm = handle->desc.K / handle->ofmblock;
      }
      /* create barrier */
      handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
      /* calculate scratch size for batchstats */
      if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
        handle->scratch_size = sizeof(float) * ( ( (size_t)handle->desc.C * (size_t)handle->desc.N ) + ( (size_t)handle->desc.C * (size_t)handle->desc.K ) );
      } else if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)  ) {
        /* Let's allocate maximum required scratch  */
        size_t size_fwd = sizeof(float) * handle->desc.K * handle->desc.N;
        /* In case of K = 1 we pad A and B to "bk=2" */
        size_t size_bwd = (handle->desc.K != 1) ? ( sizeof(float) * handle->desc.C * handle->desc.N + sizeof(libxsmm_bfloat16) * handle->desc.C * handle->desc.K ) : ( sizeof(float) * handle->desc.C * handle->desc.N + sizeof(libxsmm_bfloat16) * handle->desc.C * 2 + sizeof(libxsmm_bfloat16) * 2 * handle->desc.N );
        size_t size_upd = sizeof(float) * handle->desc.C * handle->desc.K + sizeof(libxsmm_bfloat16) * handle->desc.threads * handle->bk * handle->bc + sizeof(libxsmm_bfloat16) * (handle->desc.N * (handle->desc.C + handle->desc.K));
        handle->scratch_size = LIBXSMM_MAX(LIBXSMM_MAX(size_fwd, size_bwd), size_upd);
      } else {
        handle->scratch_size = sizeof(float) * LIBXSMM_MAX( ((size_t)handle->desc.C + (size_t)handle->desc.K) * (size_t)handle->desc.N,
            (size_t)handle->desc.C * (size_t)handle->desc.K);
      }
      /* create code pointers in some special cases */
      if ( ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) > 0) && ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED) > 0)  ) {
        if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
          float alpha = 1.0f;
          /* beta is set to 1 for ncnc kcck format because ifm is split into 2 blocks */
          float beta  = 1.0f;
          float zerobeta  = 0.0f;
          libxsmm_blasint lda = (libxsmm_blasint)handle->bk;
          libxsmm_blasint ldb = (libxsmm_blasint)handle->bc;
          libxsmm_blasint ldc = (libxsmm_blasint)handle->bk;

          if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
#ifdef ADDRESS_BRGEMM
            handle->gemm_fwd.xgemm.smra = libxsmm_smmdispatch_reducebatch_addr(handle->bk, handle->bn, handle->bc, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
            handle->gemm_bwd.xgemm.smra = libxsmm_smmdispatch_reducebatch_addr(handle->bc, handle->bn, handle->bk, &ldb, &lda, &ldb, &alpha, &beta, NULL, NULL);
#endif
#ifdef OFFSET_BRGEMM
            handle->gemm_fwd.xgemm.smro = libxsmm_smmdispatch_reducebatch_offs(handle->bk, handle->bn, handle->bc, &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
            handle->gemm_bwd.xgemm.smro = libxsmm_smmdispatch_reducebatch_offs(handle->bc, handle->bn, handle->bk, &ldb, &lda, &ldb, &alpha, &beta, NULL, NULL);
#endif
#ifdef STRIDE_BRGEMM
            handle->gemm_fwd.xgemm.smrs = libxsmm_smmdispatch_reducebatch_strd(handle->bk, handle->bn, handle->bc, handle->bk*handle->bc*sizeof(float), handle->bc*handle->bn*sizeof(float), &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
            handle->gemm_bwd.xgemm.smrs = libxsmm_smmdispatch_reducebatch_strd(handle->bc, handle->bn, handle->bk, handle->bk*handle->bc*sizeof(float), handle->bk*handle->bn*sizeof(float), &ldb, &lda, &ldb, &alpha, &beta, NULL, NULL);
            handle->gemm_bwd2.xgemm.smrs = libxsmm_smmdispatch_reducebatch_strd(handle->bc, handle->bn, handle->bk, handle->bk*handle->bc*sizeof(float), handle->bk*handle->bn*sizeof(float), &ldb, &lda, &ldb, &alpha, &zerobeta, NULL, NULL);
#endif
          } else {
            /* should not happen */
          }
        } else if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
          float alpha = 1.0f;
          float beta  = 1.0f;
          float zerobeta  = 0.0f;
          /* For UPD kernels we consider subtasking... */
          libxsmm_blasint M = handle->bk/handle->ofm_subtasks;
          libxsmm_blasint N = handle->bc/handle->ifm_subtasks;

          libxsmm_blasint lda = (libxsmm_blasint)handle->bk;
          libxsmm_blasint ldb = (libxsmm_blasint)handle->bc;
          libxsmm_blasint ldc = (libxsmm_blasint)handle->bk;

          if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
#ifdef ADDRESS_BRGEMM
            handle->gemm_fwd.xgemm.bmra = libxsmm_bmmdispatch_reducebatch_addr(handle->bk, handle->bn, handle->bc, &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
#endif
#ifdef OFFSET_BRGEMM
            handle->gemm_fwd.xgemm.bmro = libxsmm_bmmdispatch_reducebatch_offs(handle->bk, handle->bn, handle->bc, &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
#endif
#ifdef STRIDE_BRGEMM
            handle->gemm_fwd.xgemm.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd(handle->bk, handle->bn, handle->bc, handle->bk*handle->bc*sizeof(libxsmm_bfloat16), handle->bc*handle->bn*sizeof(libxsmm_bfloat16), &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
            handle->gemm_fwd2.xgemm.bmrs = libxsmm_bmmdispatch_reducebatch_strd(handle->bk, handle->bn, handle->bc, handle->bk*handle->bc*sizeof(libxsmm_bfloat16), handle->bc*handle->bn*sizeof(libxsmm_bfloat16), &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
            /* Special bwd kernels for K == 1 */
            if (handle->desc.K == 1) {
              libxsmm_blasint _bk = 2;
              handle->gemm_bwd.xgemm.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd(handle->bc, handle->bn, _bk, _bk*handle->bc*sizeof(libxsmm_bfloat16), _bk*handle->bn*sizeof(libxsmm_bfloat16), &ldb, &_bk, &ldb, &alpha, &beta, NULL, NULL);
              handle->gemm_bwd2.xgemm.bmrs = libxsmm_bmmdispatch_reducebatch_strd(handle->bc, handle->bn, _bk, _bk*handle->bc*sizeof(libxsmm_bfloat16), _bk*handle->bn*sizeof(libxsmm_bfloat16), &ldb, &_bk, &ldb, &alpha, &zerobeta, NULL, NULL);
            } else {
              handle->gemm_bwd.xgemm.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd(handle->bc, handle->bn, handle->bk, handle->bk*handle->bc*sizeof(libxsmm_bfloat16), handle->bk*handle->bn*sizeof(libxsmm_bfloat16), &ldb, &lda, &ldb, &alpha, &beta, NULL, NULL);
              handle->gemm_bwd2.xgemm.bmrs = libxsmm_bmmdispatch_reducebatch_strd(handle->bc, handle->bn, handle->bk, handle->bk*handle->bc*sizeof(libxsmm_bfloat16), handle->bk*handle->bn*sizeof(libxsmm_bfloat16), &ldb, &lda, &ldb, &alpha, &zerobeta, NULL, NULL);
            }
            lda = (libxsmm_blasint)handle->bk;
            ldb = (libxsmm_blasint)handle->bn;
            ldc = (libxsmm_blasint)handle->bk;
            handle->gemm_upd.xgemm.bsmrs = libxsmm_bsmmdispatch_reducebatch_strd(M, N, handle->bn, handle->bk*handle->bn*sizeof(libxsmm_bfloat16), handle->bc*handle->bn*sizeof(libxsmm_bfloat16), &lda, &ldb, &ldc, &alpha, &beta, NULL, NULL);
            handle->gemm_upd2.xgemm.bmrs = libxsmm_bmmdispatch_reducebatch_strd(M, N, handle->bn, handle->bk*handle->bn*sizeof(libxsmm_bfloat16), handle->bc*handle->bn*sizeof(libxsmm_bfloat16), &lda, &ldb, &ldc, &alpha, &zerobeta, NULL, NULL);
#endif
          } else {
            /* should not happen */
          }
        } else {

        }
      }
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_fullyconnected(const libxsmm_dnn_fullyconnected* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_fullyconnected*)handle);
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_fullyconnected_create_tensor_datalayout(const libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));

      if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)  || (type == LIBXSMM_DNN_INPUT)  ||
          (type == LIBXSMM_DNN_REGULAR_OUTPUT)    || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT)    ) {
        layout->format = handle->desc.buffer_format;
        if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)     || (type == LIBXSMM_DNN_INPUT)  ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
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
              *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) ) {
              layout->datatype = handle->desc.datatype_in;
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
              if (0 != layout->dim_type && 0 != layout->dim_size) {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
              }
            } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
              layout->datatype = handle->desc.datatype_out;
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
              if (0 != layout->dim_type && 0 != layout->dim_size) {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
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
        } else if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ||
              ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ||
              ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16))    ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)     || (type == LIBXSMM_DNN_INPUT)  )   {
                layout->dim_size[0] = handle->desc.C;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) )   {
                layout->dim_size[0] = handle->desc.K;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
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
        } else if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32)  && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ||
              ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16))    ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bc;
                layout->dim_size[1] = (unsigned int)handle->bn;
                layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)handle->bn;
                layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
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
      } else if ( (type == LIBXSMM_DNN_REGULAR_FILTER)  || (type == LIBXSMM_DNN_GRADIENT_FILTER)  || (type == LIBXSMM_DNN_FILTER)  ) {
        layout->format = handle->desc.filter_format;
        layout->tensor_type = LIBXSMM_DNN_FILTER;

        if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 6;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->ifmblock;
              layout->dim_size[2] = 1;
              layout->dim_size[3] = 1;
              layout->dim_size[4] = handle->blocksifm;
              layout->dim_size[5] = handle->blocksofm;
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else if ( ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) ||
              ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) )     ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_BF16;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(7*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 7;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[6] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->ifmblock/handle->fm_lp_block;
              layout->dim_size[3] = 1;
              layout->dim_size[4] = 1;
              layout->dim_size[5] = handle->blocksifm;
              layout->dim_size[6] = handle->blocksofm;
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32))   ||
              ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32))  ||
              ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16))    ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
              layout->dim_size[1] = handle->ifmblock * handle->blocksifm;
              layout->dim_size[2] = 1;
              layout->dim_size[3] = 1;
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXSMM_DNN_REGULAR_FILTER) || (type == LIBXSMM_DNN_GRADIENT_FILTER) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)handle->bc;
                layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
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
          } else if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_BF16;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 5;

              if ( (type == LIBXSMM_DNN_REGULAR_FILTER) || (type == LIBXSMM_DNN_GRADIENT_FILTER) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)2;
                layout->dim_size[1] = (unsigned int)handle->bk;
                layout->dim_size[2] = (unsigned int)handle->bc/2;
                layout->dim_size[3] = (unsigned int)(handle->desc.C / handle->bc);
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
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return layout;
}

LIBXSMM_API size_t libxsmm_dnn_fullyconnected_get_scratch_size(const libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_err_t* status) {
  size_t l_scratch_size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    l_scratch_size = handle->scratch_size + 64; /* 64 byte extra in case the user code does not care about alignment */
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXSMM_API void* libxsmm_dnn_fullyconnected_get_scratch_ptr(const libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    return handle->scratch;
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return 0;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_bind_scratch(libxsmm_dnn_fullyconnected* handle, const void* scratch) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)scratch;
  size_t offset = 0;

  if (scratch == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    /* align the internal scratch buffer if needed */
    if (address % 64 == 0) {
      handle->scratch = (void*)address;
    } else {
      offset = (64 - address % 64);
      handle->scratch = (void*)(address+offset);
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_release_scratch(libxsmm_dnn_fullyconnected* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    handle->scratch = 0;
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_bind_tensor(libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)  && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
      (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER)    ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
        handle->reg_input = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
        handle->grad_input = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
        handle->reg_output = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT ) {
        handle->grad_output = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
        handle->reg_filter = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
        handle->grad_filter = (libxsmm_dnn_tensor*)tensor;
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


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_fullyconnected_get_tensor(libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor* return_tensor = 0;

  *status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)  && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
      (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER)    ) {
    *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return return_tensor;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
      return_tensor = handle->reg_input;
    } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
      return_tensor = handle->grad_input;
    } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
      return_tensor = handle->reg_output;
    } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT ) {
      return_tensor = handle->grad_output;
    } else if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
      return_tensor = handle->reg_filter;
    } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
      return_tensor = handle->grad_filter;
    } else {
      /* cannot happen */
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return return_tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_release_tensor(libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)  && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
      (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER)    ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
      handle->reg_input = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
      handle->grad_input = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
      handle->reg_output = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT ) {
      handle->grad_output = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
      handle->reg_filter = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
      handle->grad_filter = 0;
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_execute_st(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind,
    /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  LIBXSMM_UNUSED( start_thread );
  LIBXSMM_UNUSED( tid );

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) ) {
                                             status = libxsmm_dnn_fullyconnected_st_fwd_custom( handle, start_thread, tid );
                                           } else if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED) ) {
                                             status = libxsmm_dnn_fullyconnected_st_fwd_ncnc_kcck( handle, start_thread, tid );
                                           } else {
                                             status = LIBXSMM_DNN_ERR_INVALID_FORMAT_FC;
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                           if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) ) {
                                             status = libxsmm_dnn_fullyconnected_st_bwd_custom( handle, start_thread, tid );
                                           } else if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED) ) {
                                             status = libxsmm_dnn_fullyconnected_st_bwd_ncnc_kcck( handle, start_thread, tid );
                                           } else {
                                             status = LIBXSMM_DNN_ERR_INVALID_FORMAT_FC;
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                           if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) ) {
                                             status = libxsmm_dnn_fullyconnected_st_upd_custom( handle, start_thread, tid );
                                           } else if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED) ) {
                                             status = libxsmm_dnn_fullyconnected_st_upd_ncnc_kcck( handle, start_thread, tid );
                                           } else {
                                             status = LIBXSMM_DNN_ERR_INVALID_FORMAT_FC;
                                           }
                                         } break;
      default: {
                 status = LIBXSMM_DNN_ERR_INVALID_KIND;
               }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}

