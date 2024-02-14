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
#include <libxsmm_utils.h>
#include <libxsmm.h>

void pack_c(const libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, const libxsmm_blasint C, const libxsmm_blasint K, const libxsmm_blasint bc, const libxsmm_blasint bk, const libxsmm_blasint vnni_pack) {
  const libxsmm_blasint cBlocks = C/bc;
  const libxsmm_blasint kBlocks = K/bk;
  libxsmm_blasint k1, k2, c1, c2;
  LIBXSMM_VLA_DECL(2, const libxsmm_bfloat16, real_src, src, K);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, cBlocks, bc/vnni_pack, bk, vnni_pack);

#if defined(_OPENMP)
# pragma omp parallel for private(k1,c1,c2,k2) LIBXSMM_OPENMP_COLLAPSE(2)
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

void pack_tpp_identity(const libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, const libxsmm_blasint C, const libxsmm_blasint K, const libxsmm_blasint bc, const libxsmm_blasint bk, const libxsmm_blasint vnni_pack, libxsmm_meltwfunction_unary kernel) {
  const libxsmm_blasint cBlocks = C/bc;
  const libxsmm_blasint kBlocks = K/bk;
  libxsmm_blasint k1, k2, c1, c2;
  LIBXSMM_VLA_DECL(2, const libxsmm_bfloat16, real_src, src, K);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, cBlocks, bc/vnni_pack, bk, vnni_pack);
  libxsmm_bfloat16 *const tmp = (libxsmm_bfloat16*)malloc(sizeof(libxsmm_bfloat16) * bc * bk * kBlocks);
  LIBXSMM_ASSERT(NULL != tmp);

#if defined(_OPENMP)
# pragma omp parallel for private(k1,c1,c2,k2) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, real_tmp, tmp + bc * bk * k1, bk, vnni_pack);
      libxsmm_meltw_unary_param unary_param;
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(3, real_tmp, c2/vnni_pack, k2, c2%vnni_pack, bk, vnni_pack) =
            LIBXSMM_VLA_ACCESS(2, real_src, c1*bc+c2, k1*bk+k2, K);
        }
      }
      unary_param.in.primary = (void*)&(LIBXSMM_VLA_ACCESS(3, real_tmp, 0, 0, 0, bk, vnni_pack));
      unary_param.out.primary = (void*)&(LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, 0, 0, 0, cBlocks, bc/vnni_pack, bk, vnni_pack));
      kernel( &unary_param );
    }
  }
}

void pack_tpp_normtovnni(const libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, const libxsmm_blasint C, const libxsmm_blasint K, const libxsmm_blasint bc, const libxsmm_blasint bk, const libxsmm_blasint vnni_pack, libxsmm_meltwfunction_unary kernel) {
  const libxsmm_blasint kBlocks = K/bk;
  const libxsmm_blasint cBlocks = C/bc;
  libxsmm_blasint k1, c1;
  LIBXSMM_VLA_DECL(2, const libxsmm_bfloat16, real_src, src, K);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, cBlocks, bc/vnni_pack, bk, vnni_pack);

#if defined(_OPENMP)
# pragma omp parallel for private(k1,c1) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (c1 = 0; c1 < cBlocks; c1++) {
    for (k1 = 0; k1 < kBlocks; k1++) {
      libxsmm_meltw_unary_param unary_param;
      unary_param.in.primary = (void*)&(LIBXSMM_VLA_ACCESS(2, real_src, c1*bc, k1*bk, K));
      unary_param.out.primary = (void*)&(LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, 0, 0, 0, cBlocks, bc/vnni_pack, bk, vnni_pack));
      kernel( &unary_param );
    }
  }
}

void unpack_c(const libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, const libxsmm_blasint C, const libxsmm_blasint K, const libxsmm_blasint bc, const libxsmm_blasint bk, const libxsmm_blasint vnni_pack) {
  const libxsmm_blasint cBlocks = C/bc;
  const libxsmm_blasint kBlocks = K/bk;
  libxsmm_blasint k1, k2, c1, c2;
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_dst, dst, K);
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, real_src, src, cBlocks, bc/vnni_pack, bk, vnni_pack);

#if defined(_OPENMP)
# pragma omp parallel for private(k1,c1,c2,k2) LIBXSMM_OPENMP_COLLAPSE(2)
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

void unpack_tpp_identity(const libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, const libxsmm_blasint C, const libxsmm_blasint K, const libxsmm_blasint bc, const libxsmm_blasint bk, const libxsmm_blasint vnni_pack) {
  unpack_c( src, dst, C, K, bc, bk, vnni_pack );
}

void unpack_tpp_normtovnni(const libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, const libxsmm_blasint C, const libxsmm_blasint K, const libxsmm_blasint bc, const libxsmm_blasint bk, const libxsmm_blasint vnni_pack) {
  unpack_c( src, dst, C, K, bc, bk, vnni_pack );
}

int main(int argc, char* argv[]) {
  libxsmm_blasint L =  ( argc > 1 ) ? atoi(argv[1]) :   20;
  libxsmm_blasint C =  ( argc > 2 ) ? atoi(argv[2]) : 1024;
  libxsmm_blasint K =  ( argc > 3 ) ? atoi(argv[3]) : 2048;
  libxsmm_blasint bc = ( argc > 4 ) ? atoi(argv[4]) :   32;
  libxsmm_blasint bk = ( argc > 5 ) ? atoi(argv[5]) :   32;
  libxsmm_blasint it = ( argc > 6 ) ? atoi(argv[6]) :   1000;
  libxsmm_blasint l_l, l_c, l_k;
  const unsigned int vnni_pack = libxsmm_cpuid_dot_pack_factor(LIBXSMM_DATATYPE_BF16);
  libxsmm_matdiff_info l_diff;
  double error = 0.0;
  double l_datasize = ((double)it * (double)L * (double)K * (double)C * (double)sizeof(libxsmm_bfloat16))/(1024.0*1024.0*1024.0);
  libxsmm_timer_tickint l_start;
  libxsmm_timer_tickint l_end;
  double l_total;
  libxsmm_bfloat16* l_wt_gold       = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * L * C * K, 64);
  float* l_wt_gold_f32              = (float*)           libxsmm_aligned_malloc(sizeof(float)            * L * C * K, 64);
  libxsmm_bfloat16* l_wt_packed     = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * L * C * K, 64);
  libxsmm_bfloat16* l_wt_unpack     = (libxsmm_bfloat16*)libxsmm_aligned_malloc(sizeof(libxsmm_bfloat16) * L * C * K, 64);
  float* l_wt_unpack_f32            = (float*)           libxsmm_aligned_malloc(sizeof(float)            * L * C * K, 64);
  LIBXSMM_VLA_DECL(3, float, l_wt_f32, l_wt_gold_f32, C, K);
  const libxsmm_meltw_unary_shape norm_to_vnni_shape = libxsmm_create_meltw_unary_shape( bk, bc, K, bk, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
  libxsmm_meltwfunction_unary norm_to_vnni_kernel;
  const libxsmm_meltw_unary_shape copy_shape = libxsmm_create_meltw_unary_shape( bk, bc, bk, bk, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
  libxsmm_meltwfunction_unary copy_kernel;


  libxsmm_matdiff_clear(&l_diff);

  printf("Configuration:\n");
  printf("L:%i C:%i K:%i bc:%i bk:%i vnni_pack:%i cBlocks:%i kBlocks:%i\n", L, C, K, bc, bk, vnni_pack, C/bc, K/bk );
  /* touch weights */
  for ( l_l = 0; l_l < L; l_l++) {
    for ( l_c = 0; l_c < C; l_c++) {
      for ( l_k = 0; l_k < K; l_k++ ) {
        LIBXSMM_VLA_ACCESS(3, l_wt_f32, l_l, l_c, l_k, C, K) = (float)libxsmm_rng_f64();
      }
    }
  }
  libxsmm_rne_convert_fp32_bf16( l_wt_gold_f32, l_wt_gold, L*C*K );
  libxsmm_convert_bf16_f32( l_wt_gold, l_wt_gold_f32, L*C*K );

  /* JITing kernels */
  norm_to_vnni_kernel = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, norm_to_vnni_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  copy_kernel = libxsmm_dispatch_meltw_unary( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, copy_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

  /* test dense packed */
  for ( l_l = 0; l_l < L; l_l++) {
    pack_c( l_wt_gold+(l_l*C*K), l_wt_packed+(l_l*C*K), C, K, bc, bk, vnni_pack );
  }
  for ( l_l = 0; l_l < L; l_l++) {
    unpack_c( l_wt_packed+(l_l*C*K), l_wt_unpack+(l_l*C*K), C, K, bc, bk, vnni_pack );
  }
  libxsmm_convert_bf16_f32( l_wt_unpack, l_wt_unpack_f32, L*C*K );
  libxsmm_matdiff_clear(&l_diff);
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

  /* test tpp identity pack */
  for ( l_l = 0; l_l < L; l_l++) {
    pack_tpp_identity( l_wt_gold+(l_l*C*K), l_wt_packed+(l_l*C*K), C, K, bc, bk, vnni_pack, copy_kernel );
  }
  for ( l_l = 0; l_l < L; l_l++) {
    unpack_tpp_identity( l_wt_packed+(l_l*C*K), l_wt_unpack+(l_l*C*K), C, K, bc, bk, vnni_pack );
  }
  libxsmm_convert_bf16_f32( l_wt_unpack, l_wt_unpack_f32, L*C*K );
  libxsmm_matdiff_clear(&l_diff);
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

  /* test tpp normtovnni pack */
  for ( l_l = 0; l_l < L; l_l++) {
    pack_tpp_normtovnni( l_wt_gold+(l_l*C*K), l_wt_packed+(l_l*C*K), C, K, bc, bk, vnni_pack, norm_to_vnni_kernel );
  }
  for ( l_l = 0; l_l < L; l_l++) {
    unpack_tpp_normtovnni( l_wt_packed+(l_l*C*K), l_wt_unpack+(l_l*C*K), C, K, bc, bk, vnni_pack );
  }
  libxsmm_convert_bf16_f32( l_wt_unpack, l_wt_unpack_f32, L*C*K );
  libxsmm_matdiff_clear(&l_diff);
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

  /* dense routine */
  l_start = libxsmm_timer_tick();
  for ( l_k = 0; l_k < it; l_k++ ) {
    for ( l_l = 0; l_l < L; l_l++ ) {
      pack_c( l_wt_gold+(l_l*C*K), l_wt_packed+(l_l*C*K), C, K, bc, bk, vnni_pack );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Perf. for C Pack: %fs; %f GiB/s\n", l_total, l_datasize/l_total);
  l_start = libxsmm_timer_tick();
  for ( l_k = 0; l_k < it; l_k++ ) {
    for ( l_l = 0; l_l < L; l_l++ ) {
      unpack_c( l_wt_packed+(l_l*C*K), l_wt_unpack+(l_l*C*K), C, K, bc, bk, vnni_pack );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Perf. for C unpack: %fs; %f GiB/s\n\n", l_total, l_datasize/l_total);

  /* test tpp identity pack */
  l_start = libxsmm_timer_tick();
  for ( l_k = 0; l_k < it; l_k++ ) {
    for ( l_l = 0; l_l < L; l_l++ ) {
      pack_tpp_identity( l_wt_gold+(l_l*C*K), l_wt_packed+(l_l*C*K), C, K, bc, bk, vnni_pack, copy_kernel );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Perf. for LIBXSMM identity Pack: %fs; %f GiB/s\n", l_total, l_datasize/l_total);
  l_start = libxsmm_timer_tick();
  for ( l_k = 0; l_k < it; l_k++ ) {
    for ( l_l = 0; l_l < L; l_l++ ) {
      unpack_tpp_identity( l_wt_packed+(l_l*C*K), l_wt_unpack+(l_l*C*K), C, K, bc, bk, vnni_pack );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Perf. for LIBXSMM identity unpack: %fs; %f GiB/s\n\n", l_total, l_datasize/l_total);

  /* test tpp normtovnni pack */
  l_start = libxsmm_timer_tick();
  for ( l_k = 0; l_k < it; l_k++ ) {
    for ( l_l = 0; l_l < L; l_l++ ) {
      pack_tpp_normtovnni( l_wt_gold+(l_l*C*K), l_wt_packed+(l_l*C*K), C, K, bc, bk, vnni_pack, norm_to_vnni_kernel );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Perf. for LIBXSMM norm->vnni Pack: %fs; %f GiB/s\n", l_total, l_datasize/l_total);
  l_start = libxsmm_timer_tick();
  for ( l_k = 0; l_k < it; l_k++ ) {
    for ( l_l = 0; l_l < L; l_l++ ) {
      unpack_tpp_normtovnni( l_wt_packed+(l_l*C*K), l_wt_unpack+(l_l*C*K), C, K, bc, bk, vnni_pack );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Perf. for LIBXSMM norm->vnni unpack: %fs; %f GiB/s\n\n", l_total, l_datasize/l_total);

  libxsmm_free( l_wt_unpack_f32 );
  libxsmm_free( l_wt_unpack );
  libxsmm_free( l_wt_packed );
  libxsmm_free( l_wt_gold_f32 );
  libxsmm_free( l_wt_gold );

  return 0;
}
