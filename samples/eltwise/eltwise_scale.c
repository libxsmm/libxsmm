/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


LIBXSMM_INLINE
void sfill_matrix ( float *matrix, unsigned int ld, unsigned int m, unsigned int n )
{
  unsigned int i, j;
  double dtmp;

  if ( ld < m )
  {
     fprintf(stderr,"Error is sfill_matrix: ld=%u m=%u mismatched!\n",ld,m);
     exit(EXIT_FAILURE);
  }
  for ( j = 1; j <= n; j++ )
  {
     /* Fill through the leading dimension */
     for ( i = 1; i <= ld; i++ )
     {
        dtmp = 1.0 - 2.0*libxsmm_rng_f64();
        matrix [ (j-1)*ld + (i-1) ] = (float) dtmp;
     }
  }
}

int main(int argc, char* argv[])
{
  unsigned int m = 64, n = 64, perform_scale = 1, perform_shift = 1, perform_bias = 1, scale_rows = 1, vectors_size, i, j;

  libxsmm_blasint ld_in = 64, ld_out = 64;
  float  *sinp, *sout, *scale_vals, *shift_vals, *bias_vals, *ref_out;
  unsigned short jit_flags = 0;
  libxsmm_meltwfunction_scale kernel;
  libxsmm_meltw_scale_param params;
  libxsmm_matdiff_info norms_out;

  libxsmm_init();

  libxsmm_matdiff_clear(&norms_out);

  if ( argc > 1 ) m             = atoi(argv[1]);
  if ( argc > 2 ) n             = atoi(argv[2]);
  if ( argc > 3 ) ld_in         = atoi(argv[3]);
  if ( argc > 4 ) ld_out        = atoi(argv[4]);
  if ( argc > 5 ) perform_shift = atoi(argv[5]);
  if ( argc > 6 ) perform_scale = atoi(argv[6]);
  if ( argc > 7 ) perform_bias  = atoi(argv[7]);
  if ( argc > 8 ) scale_rows    = atoi(argv[8]);

  m = LIBXSMM_MAX(m,1);
  n = LIBXSMM_MAX(n,1);
  ld_in = LIBXSMM_MAX(ld_in,(libxsmm_blasint)m);
  ld_out = LIBXSMM_MAX(ld_out,(libxsmm_blasint)m);

  vectors_size = (scale_rows == 1) ? n : m;

  /* Allocate arrays  */
  sinp      = (float*) malloc( ld_in*n*sizeof(float) );
  sout      = (float*) malloc( ld_out*n*sizeof(float) );
  ref_out   = (float*) malloc( ld_out*n*sizeof(float) );

  scale_vals = (float*) malloc(vectors_size*sizeof(float) );
  shift_vals = (float*) malloc(vectors_size*sizeof(float) );
  bias_vals  = (float*) malloc(vectors_size*sizeof(float) );

  /* Fill matrices with random data */
  sfill_matrix ( sinp, ld_in, m, n );
  sfill_matrix ( scale_vals, vectors_size, vectors_size, 1 );
  sfill_matrix ( shift_vals, vectors_size, vectors_size, 1 );
  sfill_matrix ( bias_vals, vectors_size, vectors_size, 1 );

  /* Calculate reference results...  */
  if (scale_rows == 1) {
    for (j = 0; j < n; j++) {
      float scale = scale_vals[j];
      float shift = shift_vals[j];
      float bias  = bias_vals[j];
      for (i = 0; i < m; i++) {
        float out;
        out = sinp[j*ld_in + i];
        if (perform_shift) out += shift;
        if (perform_scale) out *= scale;
        if (perform_bias)  out += bias;
        sout[j*ld_out + i] = out;
      }
    }
  } else {
    /* In this case we reduce columns */
    for (i = 0; i < m; i++) {
      float scale = scale_vals[i];
      float shift = shift_vals[i];
      float bias  = bias_vals[i];
      for (j = 0; j < n; j++) {
        float out;
        out = sinp[j*ld_in + i];
        if (perform_shift) out += shift;
        if (perform_scale) out *= scale;
        if (perform_bias)  out += bias;
        sout[j*ld_out + i] = out;
      }
    }
  }

  /* Generate JITED kernel */
  if (scale_rows == 1) {
    jit_flags = LIBXSMM_MELTW_SCALE_ROWS;
  } else {
    jit_flags = LIBXSMM_MELTW_SCALE_COLS;
  }
  if (perform_scale == 1) {
    jit_flags |=  LIBXSMM_MELTW_SCALE_MULT;
  }
  if (perform_shift == 1) {
    jit_flags |=  LIBXSMM_MELTW_SCALE_SHIFT;
  }
  if (perform_bias == 1) {
    jit_flags |=  LIBXSMM_MELTW_SCALE_ADD_BIAS;
  }

  printf("JITing scale kernel... \n");
  kernel = libxsmm_dispatch_meltw_scale(m, n, &ld_in, &ld_out, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, jit_flags);

  /* Call JITed kernel and compare result  */
  printf("Calling JITed reduce kernel... \n");
  params.in_ptr = sinp;
  params.out_ptr = sout;
  params.shift_vals_ptr = shift_vals;
  params.scale_vals_ptr = scale_vals;
  params.bias_vals_ptr  = bias_vals;
  kernel( &params );

  /* compare */
  printf("##########################################\n");
  printf("#   Correctness - Eltwise scale out      #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms_out, LIBXSMM_DATATYPE_F32, n * ld_out, 1, ref_out, sout, 0, 0);
  printf("L1 reference  : %.25g\n", norms_out.l1_ref);
  printf("L1 test       : %.25g\n", norms_out.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms_out.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms_out.l2_rel);
  printf("Linf abs.error: %.24f\n", norms_out.linf_abs);
  printf("Linf rel.error: %.24f\n", norms_out.linf_rel);
  printf("Check-norm    : %.24f\n\n", norms_out.normf_rel);

  free(sinp);
  free(sout);
  free(ref_out);
  free(scale_vals);
  free(bias_vals);
  free(shift_vals);

  return EXIT_SUCCESS;
}

