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

#include <libxsmm.h>

void pack_c(const libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, const unsigned int C, const unsigned int K, const unsigned int bc, const unsigned int bk, const unsigned int vnni_pack) {
  unsigned int k1, k2, c1, c2;
  unsigned int kBlocks = K/bk;
  unsigned int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_src, src, K);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, cBlocks, bc/vnni_pack, bk, vnni_pack);

#if defined(_OPENMP)
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, c2/vnni_pack, k2, c2%vnni_pack, cBlocks, bc/vnni_pack, bk, vnni_pack) =
            LIBXSMM_VLA_ACCESS(2, real_src, c1*bc+c2, k1*bk+k2, K);
        }
      }
    }
  }
}

void pack_tpp_identity(const libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, const unsigned int C, const unsigned int K, const unsigned int bc, const unsigned int bk, const unsigned int vnni_pack) {
  pack_c( src, dst, C, K, bc, bk, vnni_pack );
}

void pack_tpp_normtovnni(const libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, const unsigned int C, const unsigned int K, const unsigned int bc, const unsigned int bk, const unsigned int vnni_pack) {
  pack_c( src, dst, C, K, bc, bk, vnni_pack );
}

void unpack_c(const libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, const unsigned int C, const unsigned int K, const unsigned int bc, const unsigned int bk, const unsigned int vnni_pack) {
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_dst, dst, K);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_src, src, cBlocks, bc/vnni_pack, bk, vnni_pack);

#if defined(_OPENMP)
# pragma omp parallel for private(k1,c1,c2,k2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(2, real_dst, c1*bc+c2, k1*bk+k2, K) =
            LIBXSMM_VLA_ACCESS(5, real_src, k1, c1, c2/vnni_pack, k2, c2%vnni_pack, cBlocks, bc/vnni_pack, bk, vnni_pack);
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  libxsmm_blasint L =  ( argc > 1 ) ? atoi(argv[1]) : 20;
  libxsmm_blasint K =  ( argc > 2 ) ? atoi(argv[2]) : 1024;
  libxsmm_blasint C =  ( argc > 3 ) ? atoi(argv[3]) : 1024;
  libxsmm_blasint bk = ( argc > 4 ) ? atoi(argv[4]) : 32;
  libxsmm_blasint bc = ( argc > 5 ) ? atoi(argv[5]) : 32;
  libxsmm_blasint l_l, l_c, l_k;
  unsigned int vnni_pack = libxsmm_cpuid_dot_pack_factor(LIBXSMM_DATATYPE_BF16);
  libxsmm_matdiff_info l_diff;
  double error = 0.0;

  libxsmm_bfloat16* l_wt_gold       = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * L * C * K, 64);
  float* l_wt_gold_f32              = (float*)           libxsmm_aligned_malloc(sizeof(float)            * L * C * K, 64);
  libxsmm_bfloat16* l_wt_packed_c   = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * L * C * K, 64);
  libxsmm_bfloat16* l_wt_packed_1   = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * L * C * K, 64);
  libxsmm_bfloat16* l_wt_packed_2   = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * L * C * K, 64);
  libxsmm_bfloat16* l_wt_unpack     = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * L * C * K, 64);
  float* l_wt_unpack_f32            = (float*)           libxsmm_aligned_malloc(sizeof(float)            * L * C * K, 64);

  LIBXSMM_VLA_DECL(3, float, l_wt_f32, l_wt_gold_f32, C, K);

  libxsmm_matdiff_clear(&l_diff);

  /* touch weights */
  for ( l_l = 0; l_l < L; l_l++) {
    for ( l_c = 0; l_c < C; l_c++) {
      for ( l_k = 0; l_k < K; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_wt_f32, l_l, l_c, l_k, C, K) = (float)libxsmm_rng_f64();
      }
    }
  }
  libxsmm_rne_convert_fp32_bf16( l_wt_gold_f32, l_wt_gold, L*C*K );

  /* test dense packed */
  for ( l_l = 0; l_l < L; l_l++) {
    pack_c( l_wt_gold+l_l*C*K, l_wt_packed_c+l_l*C*K, C, K, bc, bk, vnni_pack );
  }
  for ( l_l = 0; l_l < L; l_l++) {
    unpack_c( l_wt_packed_c+l_l*C*K, l_wt_unpack+l_l*C+K, C, K, bc, bk, vnni_pack );
  }
  libxsmm_convert_bf16_f32( l_wt_unpack, l_wt_unpack_f32, L*C*K );
  libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, K, L*C, l_wt_unpack_f32, l_wt_gold_f32, &K, &K);
  error = libxsmm_matdiff_epsilon(&l_diff);
  printf("C Pack\n");
  printf("L1 reference  : %.25g\n", l_diff.l1_ref);
  printf("L1 test       : %.25g\n", l_diff.l1_tst);
  printf("L2 abs.error  : %.24f\n", l_diff.l2_abs);
  printf("L2 rel.error  : %.24f\n", l_diff.l2_rel);
  printf("Linf abs.error: %.24f\n", l_diff.linf_abs);
  printf("Linf rel.error: %.24f\n", l_diff.linf_rel);
  printf("Check-norm    : %.24f\n\n", error);

  /* test dense packed */
  for ( l_l = 0; l_l < L; l_l++) {
    pack_c( l_wt_gold+l_l*C*K, l_wt_packed_1+l_l*C*K, C, K, bc, bk, vnni_pack );
  }
  for ( l_l = 0; l_l < L; l_l++) {
    unpack_c( l_wt_packed_1+l_l*C*K, l_wt_unpack+l_l*C+K, C, K, bc, bk, vnni_pack );
  }
  libxsmm_convert_bf16_f32( l_wt_unpack, l_wt_unpack_f32, L*C*K );
  libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, K, L*C, l_wt_unpack_f32, l_wt_gold_f32, &K, &K);
  error = libxsmm_matdiff_epsilon(&l_diff);
  printf("LIBXSMM identity\n");
  printf("L1 reference  : %.25g\n", l_diff.l1_ref);
  printf("L1 test       : %.25g\n", l_diff.l1_tst);
  printf("L2 abs.error  : %.24f\n", l_diff.l2_abs);
  printf("L2 rel.error  : %.24f\n", l_diff.l2_rel);
  printf("Linf abs.error: %.24f\n", l_diff.linf_abs);
  printf("Linf rel.error: %.24f\n", l_diff.linf_rel);
  printf("Check-norm    : %.24f\n\n", error);

  /* test dense packed */
  for ( l_l = 0; l_l < L; l_l++) {
    pack_c( l_wt_gold+l_l*C*K, l_wt_packed_2+l_l*C*K, C, K, bc, bk, vnni_pack );
  }
  for ( l_l = 0; l_l < L; l_l++) {
    unpack_c( l_wt_packed_2+l_l*C*K, l_wt_unpack+l_l*C+K, C, K, bc, bk, vnni_pack );
  }
  libxsmm_convert_bf16_f32( l_wt_unpack, l_wt_unpack_f32, L*C*K );
  libxsmm_matdiff(&l_diff, LIBXSMM_DATATYPE_F32, K, L*C, l_wt_unpack_f32, l_wt_gold_f32, &K, &K);
  error = libxsmm_matdiff_epsilon(&l_diff);
  printf("LIBXSMM norm->vnni\n");
  printf("L1 reference  : %.25g\n", l_diff.l1_ref);
  printf("L1 test       : %.25g\n", l_diff.l1_tst);
  printf("L2 abs.error  : %.24f\n", l_diff.l2_abs);
  printf("L2 rel.error  : %.24f\n", l_diff.l2_rel);
  printf("Linf abs.error: %.24f\n", l_diff.linf_abs);
  printf("Linf rel.error: %.24f\n", l_diff.linf_rel);
  printf("Check-norm    : %.24f\n\n", error);

#if 0
  /* dense routine */
  l_start = libxsmm_timer_tick();
  for ( l_l = 0; l_l < L; l_l++) {
    pack_c( l_wt_gold, l_wt_packed_c, C, K, bc, bk, vnni_pack );
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("%fs for pack_c\n", l_total);
  printf("%f GiB/s for pack_c\n", ((double)((double)L * (double)C * (double)C * 2.0) / (l_total * 1024 * 1024 * 1024));
#endif

  libxsmm_free( l_wt_unpack_f32 );
  libxsmm_free( l_wt_unpack );
  libxsmm_free( l_wt_packed_2 );
  libxsmm_free( l_wt_packed_1 );
  libxsmm_free( l_wt_packed_c );
  libxsmm_free( l_wt_gold_f32 );
  libxsmm_free( l_wt_gold );

  return 0;
}

