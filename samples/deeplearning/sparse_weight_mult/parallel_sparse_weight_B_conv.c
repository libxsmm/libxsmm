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
#include <libxsmm.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void BlockSpMatStep1(int K, int C, int KB, int CB, int num_blocks,
                     unsigned int *colptr, unsigned int *rowidx, unsigned int *b_colptr[],
                     int *nnzb) {
    int blk_idx, i, k;

    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        nnzb[blk_idx] = 0;
        for (i = 0; i <= KB; ++i) {
            b_colptr[blk_idx][i] = 0;
        }
    }
    for (k = 0; k < K; ++k) {
        int k_blk_idx = k / KB;
        int k_blk_offset = k % KB;
        unsigned colstart = colptr[k];
        unsigned colend = colptr[k + 1];
        for (i = colstart; i < (int)colend; ++i) {
            int c = rowidx[i];
            int c_blk_idx = c / CB;
            blk_idx = k_blk_idx * C / CB + c_blk_idx;
            nnzb[blk_idx]++;
            b_colptr[blk_idx][k_blk_offset + 1]++;
        }
    }
    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        for (i = 0; i < KB; ++i) {
            b_colptr[blk_idx][i + 1] += b_colptr[blk_idx][i];
        }
    }
}

void BlockSpMatStep2(int K, int C, int KB, int CB, int num_blocks,
                     unsigned int *colptr, unsigned int *rowidx, float *values,
                     unsigned int *b_colptr[], unsigned int *b_rowidx[],
                     float *b_values[]) {
    int blk_idx, k, i;
    for (k = 0; k < K; ++k) {
        int k_blk_idx = k / KB;
        int k_blk_offset = k % KB;
        unsigned colstart = colptr[k];
        unsigned colend = colptr[k + 1];
        for (i = colstart; i < (int)colend; ++i) {
            int c = rowidx[i];
            int c_blk_idx = c / CB;
            int c_blk_offset = c % CB;
            blk_idx = k_blk_idx * C / CB + c_blk_idx;
            b_rowidx[blk_idx][b_colptr[blk_idx][k_blk_offset]] = c_blk_offset;
            b_values[blk_idx][b_colptr[blk_idx][k_blk_offset]] = values[i];
            b_colptr[blk_idx][k_blk_offset]++;
        }
    }

    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        for (i = KB; i > 0; --i) {
            b_colptr[blk_idx][i] = b_colptr[blk_idx][i - 1];
        }
        b_colptr[blk_idx][0] = 0;
    }
}

int main(int argc, char **argv) {
    int N   = (argc > 1) ? atoi(argv[1]) : 32;
    int H   = (argc > 2) ? atoi(argv[2]) : 14;
    int W   = (argc > 3) ? atoi(argv[3]) : 14;
    int C   = (argc > 4) ? atoi(argv[4]) : 256;
    int K   = (argc > 5) ? atoi(argv[5]) : 256;
    int R   = (argc > 6) ? atoi(argv[6]) : 1;
    int S   = (argc > 7) ? atoi(argv[7]) : 1;
    int padh = (argc > 8) ? atoi(argv[8]) : 0;
    int padw = (argc > 9) ? atoi(argv[9]) : 0;
    int sh  = (argc > 10) ? atoi(argv[10]) : 1;
    int sw  = (argc > 11) ? atoi(argv[11]) : 1;
    int NB  = (argc > 12) ? atoi(argv[12]) : 32;
    int CB  = (argc > 13) ? atoi(argv[13]) : 32;
    int KB  = (argc > 14) ? atoi(argv[14]) : 32;
    int nb  = (argc > 15) ? atoi(argv[15]) : 16;
    double sparse_frac = (argc > 16) ? atof(argv[16]) : 0.90;
    unsigned int REPS = (argc > 17) ? atoi(argv[17]) : 10;
    if ( CB > C ) {
      CB = C;
    }
    if ( KB > K ) {
      KB = K;
    }
    if (N < NB ||
        H < 1  ||
        W < 1  ||
        K < KB ||
        C < CB ||
        R < 1  ||
        S < 1  ||
        sh < 1 ||
        sw < 1 ||
        padh < 0 ||
        padw < 0 ||
        NB < nb ||
        C % CB != 0 ||
        N % NB != 0 ||
        nb % 16 != 0 ||
        NB % nb != 0 ||
        sparse_frac <= 0.0 ||
        sparse_frac >= 1.0 ||
        REPS <= 0) {
      return -1;
    }
    int l_n, l_h, l_w, l_p, l_q, l_c, l_r, l_s, l_nn, l_cc, l_nnn, l_k, l_kk, blk_idx;
    int i, k, n, c;
    int P = (H + 2*padh - R)/sh + 1;
    int Q = (W + 2*padw - S)/sw + 1;

    libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
    int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    float *l_A = (float *)libxsmm_aligned_malloc(sizeof(float) * N * H * W * C, 64);
    float *l_B = (float *)libxsmm_aligned_malloc(sizeof(float) * R * S * C * K, 64);
    float *l_C = (float *)libxsmm_aligned_malloc(sizeof(float) * N * P * Q * K, 64);
    float *l_C_gold =
        (float *)libxsmm_aligned_malloc(sizeof(float) * N * P * Q * K, 64);
    LIBXSMM_VLA_DECL(7, float, l_p_A, l_A, H, W, C / CB, NB / nb, CB, nb);
    LIBXSMM_VLA_DECL(4, float, l_p_B, l_B, S, K, C);
    LIBXSMM_VLA_DECL(7, float, l_p_C, l_C, P, Q, K / KB, NB / nb, KB, nb);
    LIBXSMM_VLA_DECL(7, float, l_p_C_gold, l_C_gold, P, Q, K / KB, NB / nb, KB, nb);

    /* print sizes */
    printf("Sparse Convolution kernel FWD\n");
    printf("N: %i, H:%i, W: %i, C: %i, K: %i, P: %i, Q: %i, R: %i, S: %i, sh: %i, sw: %i, padh: %i, padw: %i\n", N, H, W, C, K, P, Q, R, S, sh, sw, padh, padw);

    /* touch A */
    for (l_n = 0; l_n < N / NB; ++l_n) {
      for (l_h = 0; l_h < H; ++l_h) {
        for (l_w = 0; l_w < W; ++l_w) {
          for (l_c = 0; l_c < C / CB; ++l_c) {
            for (l_nn = 0; l_nn < NB / nb; ++l_nn) {
              for (l_cc = 0; l_cc < CB; ++l_cc) {
                for (l_nnn = 0; l_nnn < nb; ++l_nnn) {
                  LIBXSMM_VLA_ACCESS(7, l_p_A, l_n, l_h, l_w, l_c, l_nn, l_cc,
                                             l_nnn, H, W, C / CB, NB / nb, CB, nb) =
                              (float)libxsmm_rng_f64();
                }
              }
            }
          }
        }
      }
    }
    /* touch dense B and init sparse B*/
    /* we use [R][S][K][C] layout */
    int nnz = 0;
    int *rsnnz = (int*)libxsmm_aligned_malloc( R * S * sizeof(int), 64 );
    unsigned int *colptr = (unsigned int *)libxsmm_aligned_malloc(
        R * S * (K + 1) * sizeof(unsigned int), 64);
    LIBXSMM_VLA_DECL(3, unsigned int, l_p_colptr, colptr, S, K+1);
    LIBXSMM_VLA_DECL(2,          int, l_p_rsnnz, rsnnz, S);

    /* fill in dense and global CSC */
    for (l_r = 0; l_r < R; l_r++) {
      for (l_s = 0; l_s < S; l_s++) {
        LIBXSMM_VLA_ACCESS(3, l_p_colptr, l_r, l_s, 0, S, K+1) = 0;
        LIBXSMM_VLA_ACCESS(2, l_p_rsnnz, l_r, l_s, S) = 0;
        for (l_k = 0; l_k < K; l_k++) {
          LIBXSMM_VLA_ACCESS(3, l_p_colptr, l_r, l_s, l_k+1, S, K+1) = 0;
          for (l_c = 0; l_c < C; l_c++) {
            double tmp = libxsmm_rng_f64();
            if (tmp < sparse_frac) {
              tmp = 0.0;
            } else {
              LIBXSMM_VLA_ACCESS(2, l_p_rsnnz, l_r, l_s, S) = LIBXSMM_VLA_ACCESS(2, l_p_rsnnz, l_r, l_s, S) + 1;
              LIBXSMM_VLA_ACCESS(3, l_p_colptr, l_r, l_s, l_k+1, S, K+1) = LIBXSMM_VLA_ACCESS(3, l_p_colptr, l_r, l_s, l_k+1, S, K+1)+1;
              nnz++;
            }
            LIBXSMM_VLA_ACCESS(4, l_p_B, l_r, l_s, l_k, l_c, S, K, C) = (float)tmp;
          }
        }
        for (l_k = 0; l_k < K; l_k++) {
          LIBXSMM_VLA_ACCESS(3, l_p_colptr, l_r, l_s, l_k+1, S, K+1) += LIBXSMM_VLA_ACCESS(3, l_p_colptr, l_r, l_s, l_k, S, K+1);
        }
      }
    }

    /* alloc sparse structure */
    unsigned int *rowidx =
        (unsigned int *)libxsmm_aligned_malloc(nnz * sizeof(unsigned int), 64);
    float *values = (float *)libxsmm_aligned_malloc(nnz * sizeof(float), 64);
    int proc_nnz = 0;
    for (l_r = 0; l_r < R; ++l_r) {
      for (l_s = 0; l_s < S; ++l_s) {
        unsigned int *l_rowidx = rowidx + proc_nnz;
        float* l_values = values + proc_nnz;
        for (l_k = 0; l_k < K; l_k++) {
          int offset = LIBXSMM_VLA_ACCESS(3, l_p_colptr, l_r, l_s, l_k, S, K+1);
          for (l_c = 0; l_c < C; l_c++) {
            if (LIBXSMM_VLA_ACCESS(4, l_p_B, l_r, l_s, l_k, l_c, S, K, C) != 0) {
                l_rowidx[offset] = l_c;
                l_values[offset] = LIBXSMM_VLA_ACCESS(4, l_p_B, l_r, l_s, l_k, l_c, S, K, C);
                offset++;
            }
          }
        }
        proc_nnz += LIBXSMM_VLA_ACCESS(2, l_p_rsnnz, l_r, l_s, S);
      }
    }

    unsigned num_k_blocks = K / KB;
    unsigned num_c_blocks = C / CB;
    int num_blocks = num_k_blocks * num_c_blocks;
    int RSnum_blocks = R * S * num_blocks;
    unsigned int **b_colptr = (unsigned int **)libxsmm_aligned_malloc(
        RSnum_blocks * sizeof(unsigned int *), 64);
    unsigned int **b_rowidx = (unsigned int **)libxsmm_aligned_malloc(
        RSnum_blocks * sizeof(unsigned int *), 64);
    float **b_values =
        (float **)libxsmm_aligned_malloc( RSnum_blocks * sizeof(float *), 64);
    int *nnzb = (int *)libxsmm_aligned_malloc( RSnum_blocks * sizeof(int), 64);

    for (blk_idx = 0; blk_idx < RSnum_blocks; ++blk_idx) {
      b_colptr[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
        (KB + 1) * sizeof(unsigned int), 64);
    }

    int offset = 0;
    proc_nnz = 0;
    for (l_r = 0; l_r < R; ++l_r) {
      for (l_s = 0; l_s < S; ++l_s) {
        offset = (l_r*S)+l_s;
        BlockSpMatStep1(K, C, KB, CB, num_blocks, &LIBXSMM_VLA_ACCESS(3, l_p_colptr, l_r, l_s, 0, S, K+1), rowidx + proc_nnz, b_colptr+(offset*num_blocks), nnzb+(offset*num_blocks));
        proc_nnz += LIBXSMM_VLA_ACCESS(2, l_p_rsnnz, l_r, l_s, S);
      }
    }

    for (blk_idx = 0; blk_idx < RSnum_blocks; ++blk_idx) {
        b_rowidx[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
            nnzb[blk_idx] * sizeof(unsigned int), 64);
        b_values[blk_idx] =
            (float *)libxsmm_aligned_malloc(nnzb[blk_idx] * sizeof(float), 64);
    }

    offset = 0;
    proc_nnz = 0;
    for (l_r = 0; l_r < R; ++l_r) {
      for (l_s = 0; l_s < S; ++l_s) {
        offset = (l_r*S)+l_s;
        BlockSpMatStep2(K, C, KB, CB, num_blocks, &LIBXSMM_VLA_ACCESS(3, l_p_colptr, l_r, l_s, 0, S, K+1), rowidx + proc_nnz, values + proc_nnz,
                        b_colptr+(offset*num_blocks), b_rowidx+(offset*num_blocks), b_values+(offset*num_blocks));
        proc_nnz += LIBXSMM_VLA_ACCESS(2, l_p_rsnnz, l_r, l_s, S);
      }
    }

    /* touch C */
    for (l_n = 0; l_n < N / NB; ++l_n) {
      for (l_p = 0; l_p < P; ++l_p) {
        for (l_q = 0; l_q < Q; ++l_q) {
          for (l_k = 0; l_k < K / KB; ++l_k) {
            for (l_nn = 0; l_nn < NB / nb; ++l_nn) {
              for (l_kk = 0; l_kk < KB; ++l_kk) {
                for (l_nnn = 0; l_nnn < nb; ++l_nnn) {
                  LIBXSMM_VLA_ACCESS(7, l_p_C_gold, l_n, l_p, l_q, l_k, l_nn, l_kk,
                                         l_nnn, P, Q, K / KB, NB / nb, KB, nb) =
                            0.0f;
                  LIBXSMM_VLA_ACCESS(7, l_p_C, l_n, l_p, l_q, l_k, l_nn, l_kk,
                                           l_nnn, P, Q, K / KB, NB / nb, KB, nb) =
                            0.0f;
                }
              }
            }
          }
        }
      }
    }

    /* dense routine */
    for (l_n = 0; l_n < N / NB; ++l_n) {
      for (l_p = 0; l_p < P; ++l_p) {
        l_h = l_p*sh - padh;
        for (l_q = 0; l_q < Q; ++l_q) {
          l_w = l_q*sw - padw;
          for (l_k = 0; l_k < K / KB; ++l_k) {
            for (l_c = 0; l_c < C / CB; ++l_c) {
              for (l_r = 0; l_r < R; ++l_r) {
                if ( l_h+l_r < 0 || l_h+l_r >= H ) continue;
                for (l_s = 0; l_s < S; ++l_s) {
                  if ( l_w+l_s < 0 || l_w+l_s >= W ) continue;
                  for (l_nn = 0; l_nn < NB / nb; ++l_nn) {
                    for (l_kk = 0; l_kk < KB; ++l_kk) {
                      k = l_k * KB + l_kk;
                      for (l_cc = 0; l_cc < CB; ++l_cc) {
                        c = l_c * CB + l_cc;
                        for (l_nnn = 0; l_nnn < nb; ++l_nnn) {
                          LIBXSMM_VLA_ACCESS(7, l_p_C_gold, l_n, l_p, l_q, l_k,
                            l_nn, l_kk, l_nnn, P, Q, K / KB, NB / nb, KB, nb) +=
                            LIBXSMM_VLA_ACCESS(7, l_p_A, l_n, l_h+l_r, l_w+l_s, l_c,
                              l_nn, l_cc, l_nnn, H, W, C / CB, NB / nb, CB, nb) *
                            LIBXSMM_VLA_ACCESS(4, l_p_B, l_r, l_s, k, c, S, K, C);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

    /* FWD */
    float alpha = 1.0;
    float beta = 1.0;
    libxsmm_descriptor_blob l_xgemm_blob;
    libxsmm_gemm_descriptor **l_xgemm_desc =
        (libxsmm_gemm_descriptor **)libxsmm_aligned_malloc(
            RSnum_blocks * sizeof(libxsmm_gemm_descriptor *), 64);
    libxsmm_smmfunction *mykernel =
        (libxsmm_smmfunction *)libxsmm_aligned_malloc(
            RSnum_blocks * sizeof(libxsmm_smmfunction), 64);
    for (blk_idx = 0; blk_idx < RSnum_blocks; ++blk_idx) {
        l_xgemm_desc[blk_idx] = libxsmm_gemm_descriptor_dinit(
            &l_xgemm_blob, LIBXSMM_GEMM_PRECISION(float), NB / nb, KB, CB, CB,
            0, KB, alpha, beta, flags, prefetch);
        mykernel[blk_idx] =
            libxsmm_create_packed_spxgemm_csc(l_xgemm_desc[blk_idx], nb, b_colptr[blk_idx],
                                    b_rowidx[blk_idx],
                                    (const void *)b_values[blk_idx]).smm;
    }

#ifdef _OPENMP
#   pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(4) private(k,n,c,l_p,l_q,l_h,l_w,l_r,l_s)
#endif
    for (k = 0; k < K / KB; ++k) {
      for (n = 0; n < N / NB; ++n) {
        for (l_p = 0; l_p < P; ++l_p) {
          for (l_q = 0; l_q < Q; ++l_q) {
            l_h = l_p*sh - padh;
            l_w = l_q*sw - padw;
            for (l_r = 0; l_r < R; ++l_r) {
              if ( l_h+l_r < 0 || l_h+l_r >= H ) continue;
              for (l_s = 0; l_s < S; ++l_s) {
                if ( l_w+l_s < 0 || l_w+l_s >= W ) continue;
                for (c = 0; c < C / CB; ++c) {
                  mykernel[l_r * S * (K/KB) * (C/CB) +  l_s * (K/KB * C/CB) +  k * (C/CB) + c](&(LIBXSMM_VLA_ACCESS(7, l_p_A, n, l_h+l_r, l_w+l_s, c, 0, 0, 0, H, W, C / CB, NB / nb, CB, nb)),
                                               b_values[l_r * S * (K/KB) * (C/CB) +  l_s * (K/KB * C/CB) +  k * (C/CB) + c],
                                             &(LIBXSMM_VLA_ACCESS(7, l_p_C, n, l_p, l_q, k, 0, 0, 0, P, Q, K / KB, NB / nb, KB, nb)) );
                }
              }
            }
          }
        }
      }
    }
    /* check error */
    float l_max_error = 0.0f;
    for (i = 0; i < N * P * Q * K; ++i) {
        if (fabs(l_C[i] - l_C_gold[i]) > l_max_error) {
            l_max_error = (float)fabs(l_C[i] - l_C_gold[i]);
        }
    }
    printf("max error = %f\n", l_max_error);

    /* check performace */
    unsigned long long l_start = libxsmm_timer_tick();
    for (i = 0; i < (int)REPS; ++i) {
#ifdef _OPENMP
#       pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(4) private(k,n,c,l_p,l_q,l_h,l_w,l_r,l_s)
#endif
      for (k = 0; k < K / KB; ++k) {
        for (n = 0; n < N / NB; ++n) {
          for (l_p = 0; l_p < P; ++l_p) {
            for (l_q = 0; l_q < Q; ++l_q) {
              l_h = l_p*sh - padh;
              l_w = l_q*sw - padw;
              for (l_r = 0; l_r < R; ++l_r) {
                if ( l_h+l_r < 0 || l_h+l_r >= H ) continue;
                for (l_s = 0; l_s < S; ++l_s) {
                  if ( l_w+l_s < 0 || l_w+l_s >= W ) continue;
                  for (c = 0; c < C / CB; ++c) {
                    mykernel[l_r * S * (K/KB) * (C/CB) +  l_s * (K/KB * C/CB) +  k * (C/CB) + c](&(LIBXSMM_VLA_ACCESS(7, l_p_A, n, l_h+l_r, l_w+l_s, c, 0, 0, 0, H, W, C / CB, NB / nb, CB, nb)),
                                               b_values[l_r * S * (K/KB) * (C/CB) +  l_s * (K/KB * C/CB) +  k * (C/CB) + c],
                                             &(LIBXSMM_VLA_ACCESS(7, l_p_C, n, l_p, l_q, k, 0, 0, 0, P, Q, K / KB, NB / nb, KB, nb)) );
                  }
                }
              }
            }
          }
        }
      }
    }
    unsigned long long l_end = libxsmm_timer_tick();
    double l_total = libxsmm_timer_duration(l_start, l_end);
    printf("%fs for sparse (asm)\n", l_total);
    printf("%f GFLOPS for sparse (asm)\n",
           ((double)((double)REPS * (double)N * (double)P * (double)Q * (double)C * (double)R * (double)S *(double)K) * 2.0) /
               (l_total * 1.0e9));
    /* clean up */
    libxsmm_free(l_A);
    libxsmm_free(l_B);
    libxsmm_free(l_C);
    libxsmm_free(l_C_gold);
    for (blk_idx = 0; blk_idx <  RSnum_blocks; ++blk_idx) {
        libxsmm_free(b_values[blk_idx]);
        libxsmm_free(b_colptr[blk_idx]);
        libxsmm_free(b_rowidx[blk_idx]);
    }
    libxsmm_free(b_values);
    libxsmm_free(b_colptr);
    libxsmm_free(b_rowidx);
    return 0;
}
