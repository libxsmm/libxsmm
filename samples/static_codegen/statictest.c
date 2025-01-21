/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "libstatictest/libstatictest.h"

void test(unsigned int m, unsigned int n, unsigned int k, unsigned int dynld) {
  float A[m*k];
  float B[k*n];
  float C_gold[m*n];
  float C_asm[m*n];
  unsigned int lm, ln, lk;
  unsigned int error = 0;
  long long l_lda = (long long)m;
  long long l_ldb = (long long)k;
  long long l_ldc = (long long)m;
  libxsmm_matrix_op_arg myoparg;
  libxsmm_matrix_arg matrix_a;
  libxsmm_matrix_arg matrix_b;
  libxsmm_matrix_arg matrix_c;
  libxsmm_gemm_param param;

  /* init data */
  for ( lk = 0; lk < k; ++lk ) {
    for ( lm = 0; lm < m; ++lm ) {
      A[lk * m + lm] = (float)(lk * m + lm)/(float)(m*k);
    }
  }
  for ( ln = 0; ln < n; ++ln ) {
    for ( lk = 0; lk < k; ++lk ) {
      B[ln * k + lk] = (float)(ln * k + lk)/(float)(k*n);
    }
  }
  for ( ln = 0; ln < n; ++ln ) {
    for ( lm = 0; lm < m; ++lm ) {
      C_gold[ln * m + lm] = 0.0f;
      C_asm[ln * m + lm] = 0.0f;
    }
  }

  /* compute gold */
  for ( ln = 0; ln < n; ++ln ) {
    for ( lk = 0; lk < k; ++lk ) {
      for ( lm = 0; lm < m; ++lm ) {
        C_gold[ln * m + lm] += A[lk * m + lm] *  B[ln * k + lk];
      }
    }
  }

  memset( &myoparg, 0, sizeof(libxsmm_matrix_op_arg) );
  memset( &matrix_a, 0, sizeof(libxsmm_matrix_arg) );
  memset( &matrix_b, 0, sizeof(libxsmm_matrix_arg) );
  memset( &matrix_c, 0, sizeof(libxsmm_matrix_arg) );
  memset( &param, 0, sizeof(libxsmm_gemm_param) );

  matrix_a.primary = A;
  matrix_b.primary = B;
  matrix_c.primary = C_asm;
  if ( dynld != 0 ) {
    matrix_a.quinary = &l_lda;
    matrix_b.quinary = &l_ldb;
    matrix_c.quinary = &l_ldc;
  }
  param.op = myoparg;
  param.a = matrix_a;
  param.b = matrix_b;
  param.c = matrix_c;

  if ( m == 32 && n == 32 && k == 32 && dynld == 0 ) {
    printf("test one...\n");
    one( &param );
    printf("...done\n");
  } else if ( m == 64 && n == 64 && k == 64 && dynld == 0 ) {
    printf("test two...\n");
    two( &param );
    printf("...done\n");
  } else if ( m == 32 && n == 32 && k == 32 && dynld != 0 ) {
    printf("test three...\n");
    three( &param );
    printf("...done\n");
  } else if ( m == 64 && n == 64 && k == 64 && dynld != 0 ) {
    printf("test four...\n");
    four( &param );
    printf("...done\n");
  } else {
    printf("Attempting to run a case that was not pregenerated!");
    exit(-1);
  }

  for ( ln = 0; ln < n; ++ln ) {
    for ( lm = 0; lm < m; ++lm ) {
      if ( C_gold[ln * m + lm] -  C_asm[ln * m + lm] > 0.00001 ) {
        printf("(%f,%f) ", C_gold[ln * m + lm], C_asm[ln * m + lm]);
        error = 1;
      }
    }
  }
  if ( error == 0 ) {
    printf("SUCCESS\n");
  }
}

int main( int argc, char* argv[] ) {
  /* run the statically generated tests */
  test( 32, 32, 32, 0 );
  test( 64, 64, 64, 0 );
  test( 32, 32, 32, 1 );
  test( 64, 64, 64, 1 );
}

