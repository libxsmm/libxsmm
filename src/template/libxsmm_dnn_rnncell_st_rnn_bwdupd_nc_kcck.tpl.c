/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/

/* helper variables */
libxsmm_blasint i, ik, ikb, in, inb, ic, icb, jk, jb/*jn shadows global variable*/, jc, ek, en, ec, BF, KB_BLOCKS, KB;
/* tensor dimensions */
libxsmm_blasint K = handle->desc.K;
libxsmm_blasint N = handle->desc.N;
libxsmm_blasint C = handle->desc.C;
libxsmm_blasint t = handle->T;
libxsmm_blasint bk = handle->bk;
libxsmm_blasint bn = handle->bn;
libxsmm_blasint bc = handle->bc;
/* tensor raw pointers */
element_input_type  *xt  = (element_input_type* )handle->xt->data;
element_input_type  *hpD = (element_input_type* )handle->hp->data;
element_filter_type *wtD  = (element_filter_type*)handle->wt->data;
element_filter_type *rtD  = (element_filter_type*)handle->rt->data;
element_output_type *ht  = (element_output_type*)handle->ht->data;
element_input_type  *dxt = (element_input_type*)handle->dxt->data;
element_filter_type *dwD = (element_filter_type*)handle->dw->data;
element_filter_type *drD = (element_filter_type*)handle->dr->data;
element_output_type *db  = (element_output_type*)handle->db->data;
element_output_type *dht = (element_output_type*)handle->dht->data;
element_output_type *deltat = (element_output_type*)handle->scratch_deltat;
element_input_type  *scratch_xT = (element_input_type*)handle->scratch_xT;
#if 0
element_filter_type *scratch_wT = (element_filter_type*)handle->scratch_wT;
element_filter_type *scratch_rT = (element_filter_type*)handle->scratch_rT;
#endif
element_output_type *scratch_hT = (element_output_type*)handle->scratch_hT;
/* Auxiliary variables for bact-reduce calls  */
libxsmm_blasint nBlocks = N/bn;
libxsmm_blasint cBlocks = C/bc;
libxsmm_blasint kBlocks = K/bk;
unsigned long long blocks;
const float beta = 0.0;
/* multidimensional arrays */
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
LIBXSMM_VLA_DECL(4, element_filter_type, wT, wtD, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, rT, rtD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(3, element_output_type, h, ht, N, K);
LIBXSMM_VLA_DECL(3, element_input_type,  dx, dxt, N, C);
LIBXSMM_VLA_DECL(4, element_filter_type, dw, dwD, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, dr, drD, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(3, element_output_type, dh, dht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, delta, deltat, N, K);
LIBXSMM_VLA_DECL(2, element_input_type,  xT, scratch_xT, N);
#if 0
LIBXSMM_VLA_DECL(4, element_filter_type, wT, scratch_wT, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, rT, scratch_rT, kBlocks, bk, bk);
#endif
LIBXSMM_VLA_DECL(2, element_output_type, hT, scratch_hT, N);
#if defined(LIBXSMM_DNN_RNN_RELU_BWDUPD) || defined(LIBXSMM_DNN_RNN_SIGMOID_BWDUPD) || defined(LIBXSMM_DNN_RNN_TANH_BWDUPD)
element_output_type *zt = (element_output_type*)handle->internal_z;
LIBXSMM_VLA_DECL(3, element_output_type, z, zt, N, K);
#endif
/* define batch-reduce gemm kernels */
/*const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelaz = libxsmm_smmdispatch_reducebatch_addr( bc, bn, bk, &bc, &K, &C, NULL, &beta, NULL, NULL);*/
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelbz = libxsmm_smmdispatch_reducebatch_addr( bk, bk, bn, &K, &N, &bk, NULL, &beta, NULL, NULL);
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelcz = libxsmm_smmdispatch_reducebatch_addr( bk, bc, bn, &K, &N, &bk, NULL, &beta, NULL, NULL);
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelb = libxsmm_smmdispatch_reducebatch_addr( bk, bk, bn, &K, &N, &bk, NULL, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelc = libxsmm_smmdispatch_reducebatch_addr( bk, bc, bn, &K, &N, &bk, NULL, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch_addr batchreduce_kerneld = libxsmm_smmdispatch_reducebatch_addr( bk, bn, bk, &bk, &K, &K, NULL, NULL, NULL, NULL);
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernela = libxsmm_smmdispatch_reducebatch_addr( bc, bn, bk, &bc, &K, &C, NULL, NULL, NULL, NULL);

/* computing first logical thread */
const libxsmm_blasint ltid = (libxsmm_blasint)tid - (libxsmm_blasint)start_thread;

/* number of tasks that could be run in parallel for N and K blocks*/
const libxsmm_blasint work_nk = (N/bn) * (K/bk);
/* compute chunk size */
const libxsmm_blasint chunksize_nk = (work_nk % (libxsmm_blasint)handle->desc.threads == 0) ? (work_nk / (libxsmm_blasint)handle->desc.threads) : ((work_nk / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_nk = (ltid * chunksize_nk < work_nk) ? (ltid * chunksize_nk) : work_nk;
const libxsmm_blasint thr_end_nk = ((ltid + 1) * chunksize_nk < work_nk) ? ((ltid + 1) * chunksize_nk) : work_nk;

/* number of tasks that could be run in parallel for N and C blocks*/
const libxsmm_blasint work_nc = (N/bn) * (C/bc);
/* compute chunk size */
const libxsmm_blasint chunksize_nc = (work_nc % (libxsmm_blasint)handle->desc.threads == 0) ? (work_nc / (libxsmm_blasint)handle->desc.threads) : ((work_nc / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_nc = (ltid * chunksize_nc < work_nc) ? (ltid * chunksize_nc) : work_nc;
const libxsmm_blasint thr_end_nc = ((ltid + 1) * chunksize_nc < work_nc) ? ((ltid + 1) * chunksize_nc) : work_nc;

/* number of tasks that could be run in parallel for C and K blocks*/
const libxsmm_blasint work_ck = (C/bc) * (K/bk);
/* compute chunk size */
const libxsmm_blasint chunksize_ck = (work_ck % (libxsmm_blasint)handle->desc.threads == 0) ? (work_ck / (libxsmm_blasint)handle->desc.threads) : ((work_ck / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_ck = (ltid * chunksize_ck < work_ck) ? (ltid * chunksize_ck) : work_ck;
const libxsmm_blasint thr_end_ck = ((ltid + 1) * chunksize_ck < work_ck) ? ((ltid + 1) * chunksize_ck) : work_ck;

/* number of tasks that could be run in parallel for K and K blocks*/
const libxsmm_blasint work_kk = (K/bk) * (K/bk);
/* compute chunk size */
const libxsmm_blasint chunksize_kk = (work_kk % (libxsmm_blasint)handle->desc.threads == 0) ? (work_kk / (libxsmm_blasint)handle->desc.threads) : ((work_kk / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_kk = (ltid * chunksize_kk < work_kk) ? (ltid * chunksize_kk) : work_kk;
const libxsmm_blasint thr_end_kk = ((ltid + 1) * chunksize_kk < work_kk) ? ((ltid + 1) * chunksize_kk) : work_kk;

#if defined(LIBXSMM_RNN_CELL_AVX512)
int k_tasks = K/16;
int k_chunksize = (k_tasks % (libxsmm_blasint)handle->desc.threads == 0) ? (k_tasks / (libxsmm_blasint)handle->desc.threads) : ((k_tasks / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint k_thr_begin = (ltid * k_chunksize * 16 < K) ? (ltid * k_chunksize * 16) : K;
const libxsmm_blasint k_thr_end = ((ltid + 1) * k_chunksize * 16 < K) ? ((ltid + 1) * k_chunksize * 16) : K;
__m512 db_sum;
#else
/* number of tasks that could be run in parallel for K blocks*/
/* compute chunk size */
const libxsmm_blasint chunksize_k = (K % (libxsmm_blasint)handle->desc.threads == 0) ? (K / (libxsmm_blasint)handle->desc.threads) : ((K / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_k = (ltid * chunksize_k < K) ? (ltid * chunksize_k) : K;
const libxsmm_blasint thr_end_k = ((ltid + 1) * chunksize_k < K) ? ((ltid + 1) * chunksize_k) : K;

#endif

libxsmm_blasint ikic, inic, inik, icin, ikin;

/* Auxiliary arrays for batch-reduce gemm calls */
const element_filter_type *A_array[1024];
const element_output_type *B_array[1024];

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, (int)ltid);

/* Blocking reduction domain if it is too large */
BF = 1;
if (C >= 512 && K >= 512 && C%2 == 0 && K%2 == 0) {
  BF = 2;
}
if (C >= 2048 && K >= 2048 && C%8 == 0 && K%8 == 0) {
  BF = 8;
}
KB_BLOCKS = kBlocks/BF;

#if 0
if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  /* transpose W */
  for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
    ik = (ikic / (C/bc));
    ic = (ikic % (C/bc));
    for (jk = 0; jk < bk; ++jk) {
      for (jc = 0; jc < bc; ++jc) {
        LIBXSMM_VLA_ACCESS(4, wT, ic, ik, jk, jc, kBlocks, bk, bc) =  LIBXSMM_VLA_ACCESS(4, w, ik, ic, jc, jk, cBlocks, bc, bk);
      }
    }
  }
}

/* transpose R */
for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
  ik = (ikic / (K/bk));
  ic = (ikic % (K/bk));
  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bk; ++jc) {
      LIBXSMM_VLA_ACCESS(4, rT, ic, ik, jk, jc, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(4, r, ik, ic, jc, jk, kBlocks, bk, bk);
    }
  }
}
#endif

if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  /* transpose xt for current timestep */
  for (icin = thr_begin_nc; icin < thr_end_nc; ++icin ) {
    ic = (icin / (N/bn))*bc;
    in = (icin % (N/bn))*bn;

    for (jc = 0; jc < bc; ++jc) {
      for (jb = 0; jb < bn; ++jb) {
        en = in + jb;
        ec = ic + jc;
        LIBXSMM_VLA_ACCESS(2, xT, ec, en, N) =  LIBXSMM_VLA_ACCESS(3, x, t-1, en, ec, N, C);
      }
    }
  }

  /* transpose ht for current timestep */
  for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
    ik = (ikin / (N/bn))*bk;
    in = (ikin % (N/bn))*bn;

    for (jk = 0; jk < bk; ++jk) {
      for (jb = 0; jb < bn; ++jb) {
        en = in + jb;
        ek = ik + jk;
        LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) =  LIBXSMM_VLA_ACCESS(3, h, t-2, en, ek, N, K);
      }
    }
  }
}

/* The following code is for time step t-1 */
for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
  in = (inik / (K/bk))*bn;
  ik = (inik % (K/bk))*bk;

#if defined(LIBXSMM_DNN_RNN_RELU_BWDUPD)
  libxsmm_internal_matrix_relu_inverse_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, delta, t-1, in, ik, N, K) );
#endif
#if defined(LIBXSMM_DNN_RNN_SIGMOID_BWDUPD)
  libxsmm_internal_matrix_sigmoid_inverse_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, delta, t-1, in, ik, N, K) );
#endif
#if defined(LIBXSMM_DNN_RNN_TANH_BWDUPD)
  libxsmm_internal_matrix_tanh_inverse_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, delta, t-1, in, ik, N, K) );
#endif
  libxsmm_internal_matrix_inplace_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dh,    t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, delta, t-1, in, ik, N, K) );
}

libxsmm_barrier_wait(handle->barrier, (int)ltid);

if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  /* gemm kernel bwd_d */
  for (KB = 0; KB < BF; KB++) {
    for (inic = thr_begin_nc; inic < thr_end_nc; ++inic ) {
      in = (inic / (C/bc))*bn;
      icb = (inic % (C/bc));
      ic = icb * bc;
      /* Prepare arguments for batch-reduce call  */
      for (ik = 0, ikb = 0; ikb < KB_BLOCKS; ik+=bk, ikb++) {
        A_array[ikb] = &LIBXSMM_VLA_ACCESS(4, wT, icb, ikb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bc);
        B_array[ikb] = &LIBXSMM_VLA_ACCESS(3, delta, t-1, in, ik + KB*KB_BLOCKS*bk, N, K);
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, t-1, in, ic, N, C), &blocks);
    }
  }
}

if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  /* dr = delta * h^T */
  for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
    icb = ikic / (K/bk);
    ic = icb*bk;
    ikb = ikic % (K/bk);
    ik = ikb*bk;

    for (in = 0, inb = 0; in < N; in += bn, inb++) {
      A_array[inb] = &LIBXSMM_VLA_ACCESS(3, delta, t-1, in, ik, N, K);
      B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
    }
    blocks = nBlocks;
    batchreduce_kernelbz(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dr, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
  }

  /* dw = delta * x^T */
  for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
    icb = ikic / (K/bk);
    ic = icb*bc;
    ikb = ikic % (K/bk);
    ik = ikb*bk;

    for (in = 0, inb = 0; in < N; in += bn, inb++) {
      A_array[inb] = &LIBXSMM_VLA_ACCESS(3, delta, t-1, in, ik, N, K);
      B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
    }
    blocks = nBlocks;
    batchreduce_kernelcz(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dw, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
  }
}

for (i = t-2; i >= 0; --i) {
  /* let's run the cell in blocks for good locality */
  for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
    in = (inik / (K/bk))*bn;
    ikb = (inik % (K/bk));
    ik = ikb*bk;
    /* delta = dh */
    libxsmm_internal_matrix_copy_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dh, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, delta, i, in, ik, N, K) );

    /* delta += R^T * delta+1 */
    for (ic = 0; ic < kBlocks; ic++) {
      A_array[ic] = &LIBXSMM_VLA_ACCESS(4, rT, ikb, ic, 0, 0, kBlocks, bk, bk);
      B_array[ic] = &LIBXSMM_VLA_ACCESS(3, delta, i+1, in, ic*bk, N, K);
    }
    /* Reduce batch gemm call  */
    blocks = kBlocks;
    batchreduce_kerneld(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, delta, i, in, ik, N, K) , &blocks);

    /* run inverse non-linear op */
#if defined(LIBXSMM_DNN_RNN_RELU_BWDUPD)
    libxsmm_internal_matrix_relu_inverse_inplace_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, delta, i, in, ik, N, K) );
#endif
#if defined(LIBXSMM_DNN_RNN_SIGMOID_BWDUPD)
    libxsmm_internal_matrix_sigmoid_inverse_inplace_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, delta, i, in, ik, N, K) );
#endif
#if defined(LIBXSMM_DNN_RNN_TANH_BWDUPD)
    libxsmm_internal_matrix_tanh_inverse_inplace_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, delta, i, in, ik, N, K) );
#endif
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* transpose xt for current timestep */
    for (icin = thr_begin_nc; icin < thr_end_nc; ++icin ) {
      ic = (icin / (N/bn))*bc;
      in = (icin % (N/bn))*bn;

      for (jc = 0; jc < bc; ++jc) {
        for (jb = 0; jb < bn; ++jb) {
          en = in + jb;
          ec = ic + jc;
          LIBXSMM_VLA_ACCESS(2, xT, ec, en, N) =  LIBXSMM_VLA_ACCESS(3, x, i, en, ec, N, C);
        }
      }
    }

    /* transpose ht for current timestep */
    if (0 == i) {
      for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
        ik = (ikin / (N/bn))*bk;
        in = (ikin % (N/bn))*bn;

        for (jk = 0; jk < bk; ++jk) {
          for (jb = 0; jb < bn; ++jb) {
            en = in + jb;
            ek = ik + jk;
            LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) =  LIBXSMM_VLA_ACCESS(2, hp, en, ek, K);
          }
        }
      }
    } else {
      for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
        ik = (ikin / (N/bn))*bk;
        in = (ikin % (N/bn))*bn;

        for (jk = 0; jk < bk; ++jk) {
          for (jb = 0; jb < bn; ++jb) {
            en = in + jb;
            ek = ik + jk;
            LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) =  LIBXSMM_VLA_ACCESS(3, h, i-1, en, ek, N, K);
          }
        }
      }
    }
  }

  if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* dx = W^T * delta */
    for (KB = 0; KB < BF; KB++) {
      for (inic = thr_begin_nc; inic < thr_end_nc; ++inic ) {
        in = (inic / (C/bc))*bn;
        icb = (inic % (C/bc));
        ic = icb * bc;
        /* Prepare arguments for batch-reduce call  */
        for (ik = 0, ikb = 0; ikb < KB_BLOCKS; ik+=bk, ikb++) {
          A_array[ikb] = &LIBXSMM_VLA_ACCESS(4, wT, icb, ikb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bc);
          B_array[ikb] = &LIBXSMM_VLA_ACCESS(3, delta, i, in, ik + KB*KB_BLOCKS*bk, N, K);
        }
        /* Reduce batch gemm call  */
        blocks = KB_BLOCKS;
        batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, i, in, ic, N, C), &blocks);
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* dr = delta * h^T */
    for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
      icb = ikic / (K/bk);
      ic = icb*bk;
      ikb = ikic % (K/bk);
      ik = ikb*bk;

      for (in = 0, inb = 0; in < N; in += bn, inb++) {
        A_array[inb] = &LIBXSMM_VLA_ACCESS(3, delta, i, in, ik, N, K);
        B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
      }
      blocks = nBlocks;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dr, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
    }

    /* dw = delta * x^T */
    for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
      icb = ikic / (K/bk);
      ic = icb*bc;
      ikb = ikic % (K/bk);
      ik = ikb*bk;

      for (in = 0, inb = 0; in < N; in += bn, inb++) {
        A_array[inb] = &LIBXSMM_VLA_ACCESS(3, delta, i, in, ik, N, K);
        B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
      }
      blocks = nBlocks;
      batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dw, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
    }
  }
}

/* gradient bias */
if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
#if defined(LIBXSMM_RNN_CELL_AVX512)
  for (ik = k_thr_begin; ik < k_thr_end; ik += 16) {
    db_sum = _mm512_setzero_ps();
    for (i = 0; i < t; i++) {
      for (in = 0; in < N; in++) {
        db_sum = _mm512_add_ps(db_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(3, delta, i, in, ik, N, K)));
      }
    }
    LIBXSMM_INTRINSICS_MM512_STREAM_PS(&db[ik], db_sum);
  }
#else
  for (i = 0; i < t; i++) {
    for (ik = thr_begin_k; ik < thr_end_k; ik++) {
      for (in = 0; in < N; in++) {
        db[ik] += LIBXSMM_VLA_ACCESS(3, delta, i, in, ik, N, K);
      }
    }
  }
#endif
}
libxsmm_barrier_wait(handle->barrier, (int)ltid);
