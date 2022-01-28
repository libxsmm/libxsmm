/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/

/* helper variables */
libxsmm_blasint j, ik, ikb, in, inb, ic, icb, jk, jb/*jn shadows global variable*/, jc, ek, en, ec, BF, KB_BLOCKS, KB;
/* tensor dimensions */
libxsmm_blasint K = handle->desc.K;
libxsmm_blasint N = handle->desc.N;
libxsmm_blasint C = handle->desc.C;
libxsmm_blasint t = handle->T;
libxsmm_blasint bk = handle->bk;
libxsmm_blasint bn = handle->bn;
libxsmm_blasint bc = handle->bc;
libxsmm_blasint K3 = K * 3;
const libxsmm_blasint cBlocks = C/bc;
const libxsmm_blasint kBlocks = K/bk;
const libxsmm_blasint nBlocks = N/bn;
unsigned long long blocks;
/* tensor raw pointers */
element_input_type  *xt    = (element_input_type* )handle->xt->data;
element_input_type  *hpD   = (element_input_type* )handle->hp->data;
element_filter_type *w     = (element_filter_type*)handle->w->data;
element_filter_type *r     = (element_filter_type*)handle->r->data;
element_output_type *ht    = (element_output_type*)(handle->ht ? handle->ht->data : NULL);
element_output_type *it    = (element_output_type*)handle->it->data;
element_output_type *ct    = (element_output_type*)handle->cit->data;
element_output_type *ft    = (element_output_type*)handle->ft->data;
element_output_type *ot    = (element_output_type*)handle->ot->data;
element_input_type  *dxt   = (element_input_type* )handle->dxt->data;
element_input_type  *dhpD  = (element_input_type* )handle->dhp->data;
element_filter_type *dw    = (element_filter_type*)handle->dw->data;
element_filter_type *dr    = (element_filter_type*)handle->dr->data;
element_output_type *db    = (element_output_type*)handle->db->data;
element_output_type *dht   = (element_output_type*)handle->dht->data;
element_output_type *diD   = (element_output_type*)handle->scratch_di;
element_output_type *dcD   = (element_output_type*)handle->scratch_dci;
element_output_type *dfD   = (element_output_type*)handle->scratch_df;
element_output_type *doD   = (element_output_type*)handle->scratch_do;
element_output_type *doutD = (element_output_type*)handle->scratch_deltat;
element_input_type  *scratch_xT  = (element_input_type* )handle->scratch_xT;
element_filter_type *scratch_wT  = (element_filter_type*)handle->scratch_wT;
element_filter_type *scratch_rT  = (element_filter_type*)handle->scratch_rT;
element_output_type *scratch_hT  = (element_output_type*)handle->scratch_hT;
element_output_type *scratch_oT  = (element_output_type*)handle->scratch_dpB;
element_filter_type *w_scratch   = (element_filter_type*)handle->scratch_w;
element_filter_type *r_scratch   = (element_filter_type*)handle->scratch_r;
element_filter_type *wiD   = &(w[0]);
element_filter_type *wcD   = &(w[K]);
element_filter_type *wfD   = &(w[2*K]);
element_filter_type *riD   = &(r[0]);
element_filter_type *rcD   = &(r[K]);
element_filter_type *rfD   = &(r[2*K]);
element_filter_type *dwiD  = &(dw[0]);
element_filter_type *dwcD  = &(dw[K]);
element_filter_type *dwfD  = &(dw[2*K]);
element_filter_type *driD  = &(dr[0]);
element_filter_type *drcD  = &(dr[K]);
element_filter_type *drfD  = &(dr[2*K]);
element_filter_type *dwiD_scratch  = &(w_scratch[0]);
element_filter_type *dwcD_scratch  = &(w_scratch[C*K]);
element_filter_type *dwfD_scratch  = &(w_scratch[2*C*K]);
element_filter_type *driD_scratch  = &(r_scratch[0]);
element_filter_type *drcD_scratch  = &(r_scratch[K*K]);
element_filter_type *drfD_scratch  = &(r_scratch[2*K*K]);
element_output_type *dbi   = &(db[0]);
element_output_type *dbc   = &(db[K]);
element_output_type *dbf   = &(db[2*K]);
element_filter_type *scratch_wiT = &(scratch_wT[0]);
element_filter_type *scratch_wcT = &(scratch_wT[C*K]);
element_filter_type *scratch_wfT = &(scratch_wT[2*C*K]);
element_filter_type *scratch_riT = &(scratch_rT[0]);
element_filter_type *scratch_rcT = &(scratch_rT[K*K]);
element_filter_type *scratch_rfT = &(scratch_rT[2*K*K]);
element_output_type *t1D   = (element_output_type*)handle->scratch_t1;
element_output_type *t2D   = (element_output_type*)handle->scratch_t2;
/* multidimensional arrays */
LIBXSMM_VLA_DECL(2, element_output_type, t1, t1D, K);
LIBXSMM_VLA_DECL(2, element_output_type, t2, t2D, K);
LIBXSMM_VLA_DECL(3, element_input_type,  x, xt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  hp, hpD, K);
LIBXSMM_VLA_DECL(2, element_filter_type, wi, wiD, K3);
LIBXSMM_VLA_DECL(2, element_filter_type, wc, wcD, K3);
LIBXSMM_VLA_DECL(2, element_filter_type, wf, wfD, K3);
LIBXSMM_VLA_DECL(2, element_filter_type, ri, riD, K3);
LIBXSMM_VLA_DECL(2, element_filter_type, rc, rcD, K3);
LIBXSMM_VLA_DECL(2, element_filter_type, rf, rfD, K3);
LIBXSMM_VLA_DECL(3, element_output_type, h, ht, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, i, it, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, c, ct, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, f, ft, N, K);
LIBXSMM_VLA_DECL(3, element_output_type, o, ot, N, K);
LIBXSMM_VLA_DECL(3, element_input_type,  dx, dxt, N, C);
LIBXSMM_VLA_DECL(2, element_input_type,  dhp, dhpD, K);
LIBXSMM_VLA_DECL(4, element_filter_type, dwi, dwiD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, dwc, dwcD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, dwf, dwfD_scratch, cBlocks, bc, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, dri, driD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, drc, drcD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, drf, drfD_scratch, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(2, element_filter_type, dwi_ck, dwiD, K3);
LIBXSMM_VLA_DECL(2, element_filter_type, dwc_ck, dwcD, K3);
LIBXSMM_VLA_DECL(2, element_filter_type, dwf_ck, dwfD, K3);
LIBXSMM_VLA_DECL(2, element_filter_type, dri_ck, driD, K3);
LIBXSMM_VLA_DECL(2, element_filter_type, drc_ck, drcD, K3);
LIBXSMM_VLA_DECL(2, element_filter_type, drf_ck, drfD, K3);
LIBXSMM_VLA_DECL(3, element_output_type, dh, dht, N, K);
LIBXSMM_VLA_DECL(2, element_output_type, di, diD, K);
LIBXSMM_VLA_DECL(2, element_output_type, dc, dcD, K);
LIBXSMM_VLA_DECL(2, element_output_type, df, dfD, K);
LIBXSMM_VLA_DECL(2, element_output_type, dp, doD, K);
LIBXSMM_VLA_DECL(2, element_output_type, dout, doutD, K);
LIBXSMM_VLA_DECL(2, element_input_type,  xT,  scratch_xT, N);
LIBXSMM_VLA_DECL(4, element_filter_type, wiT, scratch_wiT, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, wcT, scratch_wcT, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, wfT, scratch_wfT, kBlocks, bk, bc);
LIBXSMM_VLA_DECL(4, element_filter_type, riT, scratch_riT, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rcT, scratch_rcT, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(4, element_filter_type, rfT, scratch_rfT, kBlocks, bk, bk);
LIBXSMM_VLA_DECL(2, element_output_type, hT,  scratch_hT, N);
LIBXSMM_VLA_DECL(2, element_output_type, oT,  scratch_oT, N);
element_output_type *dout_ptr = NULL;
/* define batch-reduce gemm kernels */
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernela  = libxsmm_smmdispatch_reducebatch_addr( bc, bn, bk, &bc, &K, &C,  NULL, NULL, NULL, NULL );
#if 0
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelb  = libxsmm_smmdispatch_reducebatch_addr( bk, bk, bn, &bk, &N, &bk, NULL, NULL, NULL, NULL );
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelc  = libxsmm_smmdispatch_reducebatch_addr( bk, bc, bn, &bk, &N, &bk, NULL, NULL, NULL, NULL );
#endif
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelb1 = libxsmm_smmdispatch_reducebatch_addr( bk, bk, bn, &K,  &N, &bk, NULL, NULL, NULL, NULL );
const libxsmm_smmfunction_reducebatch_addr batchreduce_kernelc1 = libxsmm_smmdispatch_reducebatch_addr( bk, bc, bn, &K,  &N, &bk, NULL, NULL, NULL, NULL );
const libxsmm_smmfunction_reducebatch_addr batchreduce_kerneld  = libxsmm_smmdispatch_reducebatch_addr( bk, bn, bk, &bk, &K, &K,  NULL, NULL, NULL, NULL );

/* Auxiliary arrays for batch-reduce gemm calls */
const element_filter_type *A_array[1024];
const element_output_type *B_array[1024];

#if 0
LIBXSMM_VLA_DECL(4, element_output_type, diB, (element_output_type*)handle->scratch_diB,  kBlocks, bn, bk);
LIBXSMM_VLA_DECL(4, element_output_type, dcB, (element_output_type*)handle->scratch_dciB, kBlocks, bn, bk);
LIBXSMM_VLA_DECL(4, element_output_type, dfB, (element_output_type*)handle->scratch_dfB,  kBlocks, bn, bk);
#endif

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

/* number of tasks that could be run in parallel for K blocks*/
/* compute chunk size */
const libxsmm_blasint chunksize_k = (K % (libxsmm_blasint)handle->desc.threads == 0) ? (K / (libxsmm_blasint)handle->desc.threads) : ((K / (libxsmm_blasint)handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const libxsmm_blasint thr_begin_k = (ltid * chunksize_k < K) ? (ltid * chunksize_k) : K;
const libxsmm_blasint thr_end_k = ((ltid + 1) * chunksize_k < K) ? ((ltid + 1) * chunksize_k) : K;

/* int bcbk_multiples_of_16 = ((bc % 16 == 0) && (bk % 16 == 0)) ? 1 : 0; */

libxsmm_blasint ikic, inic, inik, icin, ikin;

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, (int)ltid);

/* Blocking reduction domain if it is too large */
BF = 1;
if (K > 1024 && K <= 2048) {
  BF = 8;
  while (kBlocks % BF != 0) {
    BF--;
  }
}

if (K > 2048) {
  BF = 16;
  while (kBlocks % BF != 0) {
    BF--;
  }
}
KB_BLOCKS = kBlocks/BF;

/* initialization is done at the beginning */
if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  libxsmm_internal_matrix_zero(N*C*t, dxt, start_thread, tid, handle->desc.threads);
}

/* initialization is done at the beginning */
if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  libxsmm_internal_matrix_zero(C*K*3, w_scratch,  start_thread, tid, handle->desc.threads);
  libxsmm_internal_matrix_zero(K*K*3, r_scratch,  start_thread, tid, handle->desc.threads);
  libxsmm_internal_matrix_zero(K*3,   db,         start_thread, tid, handle->desc.threads);
}

/* transpose W */
for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
  ic = (ikic / (K/bk));
  ik = (ikic % (K/bk));
  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bc; ++jc) {
      LIBXSMM_VLA_ACCESS(4, wiT, ic, ik, jk, jc, kBlocks, bk, bc) =  LIBXSMM_VLA_ACCESS(2, wi, ic*bc+jc, ik*bk+jk, K3);
      LIBXSMM_VLA_ACCESS(4, wcT, ic, ik, jk, jc, kBlocks, bk, bc) =  LIBXSMM_VLA_ACCESS(2, wc, ic*bc+jc, ik*bk+jk, K3);
      LIBXSMM_VLA_ACCESS(4, wfT, ic, ik, jk, jc, kBlocks, bk, bc) =  LIBXSMM_VLA_ACCESS(2, wf, ic*bc+jc, ik*bk+jk, K3);
    }
  }
}

/* transpose R */
for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
  ik = (ikic / (K/bk));
  ic = (ikic % (K/bk));
  for (jk = 0; jk < bk; ++jk) {
    for (jc = 0; jc < bk; ++jc) {
      LIBXSMM_VLA_ACCESS(4, riT, ic, ik, jk, jc, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(2, ri, ic*bk+jc, ik*bk+jk, K3);
      LIBXSMM_VLA_ACCESS(4, rcT, ic, ik, jk, jc, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(2, rc, ic*bk+jc, ik*bk+jk, K3);
      LIBXSMM_VLA_ACCESS(4, rfT, ic, ik, jk, jc, kBlocks, bk, bk) =  LIBXSMM_VLA_ACCESS(2, rf, ic*bk+jc, ik*bk+jk, K3);
    }
  }
}
libxsmm_barrier_wait(handle->barrier, (int)ltid);

for (j = t-1; j >= 0; --j) {
  /* let's run the cell in blocks for good locality */
  for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
    in = (inik % (N/bn))*bn;
    ik = (inik / (N/bn))*bk;

    /* compute dhp */
    if (j == t-1) {
      libxsmm_internal_matrix_copy_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dh, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) );
    } else {
      libxsmm_internal_matrix_add_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dh, j,   in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) );
    }
    /* df = dout . (1 - c) . (1 - (f . f)) */
    libxsmm_internal_matrix_complement_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_square_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K),   &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, df, in, ik, K) );
    /* dc = dout . (hp - f) . c . (1 - c) */
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K),   &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    if (0 == j) {
      libxsmm_internal_matrix_sub_ld(             bk, bn, K, &LIBXSMM_VLA_ACCESS(2, hp, in, ik, K),        &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    } else {
      LIBXSMM_ASSERT(NULL != ht); /* coverity[var_deref_op] */
      libxsmm_internal_matrix_sub_ld(             bk, bn, K, &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    }
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K),   &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dc, in, ik, K) );
  }

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* transpose xt for current timestep */
    for (icin = thr_begin_nc; icin < thr_end_nc; ++icin ) {
      in = (icin / (C/bc))*bn;
      ic = (icin % (C/bc))*bc;

      for (jc = 0; jc < bc; ++jc) {
        for (jb = 0; jb < bn; ++jb) {
          en = in + jb;
          ec = ic + jc;
          LIBXSMM_VLA_ACCESS(2, xT, ec, en, N) = LIBXSMM_VLA_ACCESS(3, x, j, en, ec, N, C);
        }
      }
    }

    /* transpose ht for current timestep */
    if (j == 0) {
      for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
        in = (ikin / (K/bk))*bn;
        ik = (ikin % (K/bk))*bk;

        for (jk = 0; jk < bk; ++jk) {
          for (jb = 0; jb < bn; ++jb) {
            en = in + jb;
            ek = ik + jk;
            LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) = LIBXSMM_VLA_ACCESS(2, hp, en, ek, K);
          }
        }
      }
    } else {
      for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
        in = (ikin / (K/bk))*bn;
        ik = (ikin % (K/bk))*bk;

        for (jk = 0; jk < bk; ++jk) {
          for (jb = 0; jb < bn; ++jb) {
            en = in + jb;
            ek = ik + jk;
            LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) = LIBXSMM_VLA_ACCESS(3, h, j-1, en, ek, N, K);
          }
        }
      }
    }

    /* transpose ot for current timestep */
    for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
      in = (ikin / (K/bk))*bn;
      ik = (ikin % (K/bk))*bk;

      for (jk = 0; jk < bk; ++jk) {
        for (jb = 0; jb < bn; ++jb) {
          en = in + jb;
          ek = ik + jk;
          LIBXSMM_VLA_ACCESS(2, oT, ek, en, N) = LIBXSMM_VLA_ACCESS(3, o, j, en, ek, N, K);
        }
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);

  /* do = {R_f}^T * df */
  for (KB = 0; KB < BF; KB++) {
    for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
      in = (inik % (N/bn))*bn;
      ikb = inik / (N/bn);
      ik = ikb*bk;

      if (KB == 0) libxsmm_internal_matrix_zero_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dp, in, ik, K) );
      for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, rfT, ikb, icb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ic + KB*KB_BLOCKS*bk, K);
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kerneld(A_array, B_array, &LIBXSMM_VLA_ACCESS(2, dp, in, ik, K), &blocks);
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);

  /* di = do . hp . i . (1 - i) */
  for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
    in = (inik % (N/bn))*bn;
    ik = (inik / (N/bn))*bk;
    libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    if (0 == j) {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, hp, in, ik, K),        &LIBXSMM_VLA_ACCESS(2, dp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    } else {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    }
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, di, in, ik, K) );
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);

  if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* dx = W^T * dicf */
    for (KB = 0; KB < BF; KB++) {
      for (inic = thr_begin_nc; inic < thr_end_nc; ++inic ) {
        in = (inic % (N/bn))*bn;
        icb = inic / (N/bn);
        ic = icb*bc;

        for (ik = 0, ikb = 0; ikb < KB_BLOCKS; ik += bk, ikb++) {
          A_array[ikb] = &LIBXSMM_VLA_ACCESS(4, wiT, icb, ikb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bc);
          B_array[ikb] = &LIBXSMM_VLA_ACCESS(2, di,  in, ik + KB*KB_BLOCKS*bk, K);
        }
        /* Reduce batch gemm call  */
        blocks = KB_BLOCKS;
        batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C), &blocks);

        for (ik = 0, ikb = 0; ikb < KB_BLOCKS; ik += bk, ikb++) {
          A_array[ikb] = &LIBXSMM_VLA_ACCESS(4, wcT, icb, ikb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bc);
          B_array[ikb] = &LIBXSMM_VLA_ACCESS(2, dc,  in, ik + KB*KB_BLOCKS*bk, K);
        }
        /* Reduce batch gemm call  */
        batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C), &blocks);

        for (ik = 0, ikb = 0; ikb < KB_BLOCKS; ik += bk, ikb++) {
          A_array[ikb] = &LIBXSMM_VLA_ACCESS(4, wfT, icb, ikb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bc);
          B_array[ikb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ik + KB*KB_BLOCKS*bk, K);
        }
        /* Reduce batch gemm call  */
        batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C), &blocks);
      }
    }
  }

  for (KB = 0; KB < BF; KB++) {
    for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
      in = (inik % (N/bn))*bn;
      ikb = inik / (N/bn);
      ik = ikb*bk;
      dout_ptr = (j > 0) ? (element_output_type*) &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) : (element_output_type*) &LIBXSMM_VLA_ACCESS(2, dhp, in, ik, K);

      if (0 == KB) {
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dp,   in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
        libxsmm_internal_matrix_add_ld(          bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K),      &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), dout_ptr );
      }

      /* dhp += R^T * dic */
      for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, riT, ikb, icb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(2, di,  in, ic + KB*KB_BLOCKS*bk, K);
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kerneld(A_array, B_array, dout_ptr, &blocks);

      for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, rcT, ikb, icb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(2, dc,  in, ic + KB*KB_BLOCKS*bk, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kerneld(A_array, B_array, dout_ptr, &blocks);
    }
  }

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    if ((C == K) && (bc == bk) /*&& (bcbk_multiples_of_16 == 1)*/) {
#if 0
      if (K % 2048 != 0) {
#endif
        /* Interleave computation of dr = dicf * o^T/h^T and dw = dicf * x^T to take advantage of temporal locality */
        for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
          icb = ikic / (K/bk);
          ic = icb*bk;
          ikb = ikic % (K/bk);
          ik = ikb*bk;
          blocks = nBlocks;

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, di, in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, oT, ic, in, N);
          }
          batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, di, in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dc, in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, oT, ic, in, N);
          }
          batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dc, in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, df, in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
          }
          batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, df, in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        }
#if 0
      } else {
        /* Interleave computation of dr = dicf * o^T/h^T and dw = dicf * x^T to take advantage of temporal locality */
        /* Use blocked format for di, dc, df */
        for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
          icb = ikic / (K/bk);
          ic = icb*bk;
          ikb = ikic % (K/bk);
          ik = ikb*bk;
          blocks = nBlocks;

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, diB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, oT, ic, in, N);
          }
          batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, diB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, dcB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, oT, ic, in, N);
          }
          batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, dcB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, dfB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
          }
          batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, dfB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        }
      }
#endif
    } else {
      /* dr = dicf * o^T/h^T */
      for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
        icb = ikic / (K/bk);
        ic = icb*bk;
        ikb = ikic % (K/bk);
        ik = ikb*bk;

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, di, in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, oT, ic, in, N);
        }
        blocks = nBlocks;
        batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dc, in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, oT, ic, in, N);
        }
        batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, df, in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
        }
        batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
      }

      /* dw = dicf * x^T */
      for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
        icb = ikic / (K/bk);
        ic = icb*bc;
        ikb = ikic % (K/bk);
        ik = ikb*bk;

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, di, in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
        }
        blocks = nBlocks;
        batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dc, in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
        }
        batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
        }
        batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
      }
    }

    /* gradient bias */
    for (ik = thr_begin_k; ik < thr_end_k; ik++) {
      for (in = 0; in < N; in++) {
        dbi[ik] += LIBXSMM_VLA_ACCESS(2, di, in, ik, K);
        dbc[ik] += LIBXSMM_VLA_ACCESS(2, dc, in, ik, K);
        dbf[ik] += LIBXSMM_VLA_ACCESS(2, df, in, ik, K);
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}

if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
  /* Store result weight matrices in CK format */
  for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
    icb = ikic / (K/bk);
    ic = icb*bc;
    ikb = ikic % (K/bk);
    ik = ikb*bk;
    for (jc = 0; jc < bc; ++jc) {
      for (jk = 0; jk < bk; ++jk) {
        LIBXSMM_VLA_ACCESS(2, dwi_ck, ic+jc, ik+jk, K3) = LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, jc, jk, cBlocks, bc, bk);
        LIBXSMM_VLA_ACCESS(2, dwc_ck, ic+jc, ik+jk, K3) = LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, jc, jk, cBlocks, bc, bk);
        LIBXSMM_VLA_ACCESS(2, dwf_ck, ic+jc, ik+jk, K3) = LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, jc, jk, cBlocks, bc, bk);
      }
    }
  }

  for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
    icb = ikic / (K/bk);
    ic = icb*bk;
    ikb = ikic % (K/bk);
    ik = ikb*bk;
    for (jc = 0; jc < bk; ++jc) {
      for (jk = 0; jk < bk; ++jk) {
        LIBXSMM_VLA_ACCESS(2, dri_ck, ic+jc, ik+jk, K3) = LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, jc, jk, kBlocks, bk, bk);
        LIBXSMM_VLA_ACCESS(2, drc_ck, ic+jc, ik+jk, K3) = LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, jc, jk, kBlocks, bk, bk);
        LIBXSMM_VLA_ACCESS(2, drf_ck, ic+jc, ik+jk, K3) = LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, jc, jk, kBlocks, bk, bk);
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}
