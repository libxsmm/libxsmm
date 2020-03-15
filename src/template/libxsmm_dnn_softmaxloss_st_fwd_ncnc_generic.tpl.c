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
libxsmm_blasint bn = handle->bn;
libxsmm_blasint Bn = handle->Bn;
libxsmm_blasint bc = handle->bc;
libxsmm_blasint Bc = handle->Bc;

/* loop counters */
int i = 0;
int j = 0;
libxsmm_blasint img1, img2, ifm1, ifm2;

/* computing first logical thread */
const int ltid = tid - start_thread;

/* number of tasks that could run in parallel for the batch */
const int n_work = Bn * bn;
/* compute chunk size */
const int n_chunksize = (n_work % handle->desc.threads == 0) ? (n_work / handle->desc.threads) : ((n_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int n_thr_begin = (ltid * n_chunksize < n_work) ? (ltid * n_chunksize) : n_work;
const int n_thr_end = ((ltid + 1) * n_chunksize < n_work) ? ((ltid + 1) * n_chunksize) : n_work;

LIBXSMM_VLA_DECL(4,       element_output_type, output, (element_output_type*)handle->reg_output->data, Bc, bn, bc);
LIBXSMM_VLA_DECL(4, const element_input_type,   input,  (element_input_type*)handle->reg_input->data,  Bc, bn, bc);
LIBXSMM_VLA_DECL(2, const element_label_type,   label,  (element_label_type*)handle->label->data,              bc);

/* lazy barrier init */
libxsmm_barrier_init( handle->barrier, ltid );

for ( i = n_thr_begin; i < n_thr_end; ++i ) {
  element_output_type max =        FLT_MIN;
  element_output_type sum_of_exp = 0.0f;

  img1 = i/bn;
  img2 = i%bn;

  /* set output to input and set compute max per image */
  for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
    for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
      LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) = LIBXSMM_VLA_ACCESS( 4, input, img1, ifm1, img2, ifm2, Bc, bn, bc );
      if ( LIBXSMM_VLA_ACCESS( 4, input, img1, ifm1, img2, ifm2, Bc, bn, bc ) > max ) {
        max = LIBXSMM_VLA_ACCESS( 4, input, img1, ifm1, img2, ifm2, Bc, bn, bc );
      }
    }
  }

  /* sum exp over outputs */
  for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
    for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
      LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) = exp( LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) - max );
      sum_of_exp += LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc );
    }
  }

  /* scale output */
  sum_of_exp = 1.0f/sum_of_exp;
  for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
    for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
      LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) = LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) * sum_of_exp;
    }
  }
}

libxsmm_barrier_wait( handle->barrier, ltid );

/* calculate loss single threaded */
if ( ltid == 0 ) {
  handle->loss = 0.0f;
  for ( img1 = 0; img1 < Bn; ++img1 ) {
    for ( img2 = 0; img2 <bn; ++img2 ) {
      libxsmm_blasint ifm = (libxsmm_blasint)LIBXSMM_VLA_ACCESS( 2, label, ifm1, ifm2, bc );
      libxsmm_blasint ifm1b = ifm/bc;
      libxsmm_blasint ifm2b = ifm%bc;
      element_output_type val = ( LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1b, img2, ifm2b, Bc, bn, bc ) > FLT_MIN ) ? LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1b, img2, ifm2b, Bc, bn, bc ) : FLT_MIN;
      handle->loss = log( val );
    }
  }
  handle->loss = ((-1.0f)*handle->loss)/handle->desc.N;
}

libxsmm_barrier_wait( handle->barrier, ltid );
