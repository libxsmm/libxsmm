/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Xing Liu (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void BlockSpMatStep1(int K, int C, int KB, int CB, unsigned int *colptr,
                     unsigned int *rowidx, unsigned int *b_colptr[],
                     int *nnzb) {
    int num_blocks = K / KB * C / CB;
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

void BlockSpMatStep2(int K, int C, int KB, int CB, unsigned int *colptr,
                     unsigned int *rowidx, float *values,
                     unsigned int *b_colptr[], unsigned int *b_rowidx[],
                     float *b_values[]) {
    int num_blocks = K / KB * C / CB;
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
    int N = (argc > 1) ? atoi(argv[1]) : 2048;
    int C = (argc > 2) ? atoi(argv[2]) : 512;
    int K = (argc > 3) ? atoi(argv[3]) : 512;
    int NB = (argc > 4) ? atoi(argv[4]) : 32;
    int CB = (argc > 5) ? atoi(argv[5]) : 32;
    int KB = (argc > 6) ? atoi(argv[6]) : 32;
    int nb = (argc > 7) ? atoi(argv[7]) : 16;
    double sparse_frac = (argc > 8) ? atof(argv[8]) : 0.90;
    unsigned int REPS = (argc > 9) ? atoi(argv[9]) : 10;
    if (N < NB ||
        K < KB ||
        C < CB ||
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
    int l_n, l_c, l_nn, l_cc, l_nnn, l_k, l_kk, blk_idx;
    int i, k, n, c;

    libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
    int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    float *l_A = (float *)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);
    float *l_B = (float *)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);
    float *l_C = (float *)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
    float *l_C_gold =
        (float *)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
    LIBXSMM_VLA_DECL(5, float, l_p_A, l_A, C / CB, NB / nb, CB, nb);
    LIBXSMM_VLA_DECL(5, float, l_p_C, l_C, K / KB, NB / nb, KB, nb);
    LIBXSMM_VLA_DECL(5, float, l_p_C_gold, l_C_gold, K / KB, NB / nb, KB, 16);
    /* touch A */
    for (l_n = 0; l_n < N / NB; ++l_n) {
        for (l_c = 0; l_c < C / CB; ++l_c) {
            for (l_nn = 0; l_nn < NB / nb; ++l_nn) {
                for (l_cc = 0; l_cc < CB; ++l_cc) {
                    for (l_nnn = 0; l_nnn < nb; ++l_nnn) {
                        LIBXSMM_VLA_ACCESS(5, l_p_A, l_n, l_c, l_nn, l_cc,
                                           l_nnn, C / CB, NB / nb, CB, nb) =
                            (float)libxsmm_rng_f64();
                    }
                }
            }
        }
    }
    /* touch dense B and init sparse B*/
    int nnz = 0;
    unsigned int *colptr = (unsigned int *)libxsmm_aligned_malloc(
        (K + 1) * sizeof(unsigned int), 64);
    colptr[0] = 0;
    for (l_k = 0; l_k < K; l_k++) {
        colptr[l_k + 1] = 0;
        for (l_c = 0; l_c < C; l_c++) {
            double tmp = libxsmm_rng_f64();
            if (tmp < sparse_frac) {
                tmp = 0.0;
            } else {
                nnz++;
                colptr[l_k + 1]++;
            }
            l_B[l_k * C + l_c] = (float)tmp;
        }
    }
    for (l_k = 0; l_k < K; l_k++) {
        colptr[l_k + 1] += colptr[l_k];
    }
    unsigned int *rowidx =
        (unsigned int *)libxsmm_aligned_malloc(nnz * sizeof(unsigned int), 64);
    float *values = (float *)libxsmm_aligned_malloc(nnz * sizeof(float), 64);
    for (l_k = 0; l_k < K; l_k++) {
        int offset = colptr[l_k];
        for (l_c = 0; l_c < C; l_c++) {
            if (l_B[l_k * C + l_c] != 0) {
                rowidx[offset] = l_c;
                values[offset] = l_B[l_k * C + l_c];
                offset++;
            }
        }
    }
    unsigned num_k_blocks = K / KB;
    unsigned num_c_blocks = C / CB;
    int num_blocks = num_k_blocks * num_c_blocks;
    unsigned int **b_colptr = (unsigned int **)libxsmm_aligned_malloc(
        num_blocks * sizeof(unsigned int *), 64);
    unsigned int **b_rowidx = (unsigned int **)libxsmm_aligned_malloc(
        num_blocks * sizeof(unsigned int *), 64);
    float **b_values =
        (float **)libxsmm_aligned_malloc(num_blocks * sizeof(float *), 64);
    int *nnzb = (int *)libxsmm_aligned_malloc(num_blocks * sizeof(int), 64);
    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        b_colptr[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
            (KB + 1) * sizeof(unsigned int), 64);
    }
    BlockSpMatStep1(K, C, KB, CB, colptr, rowidx, b_colptr, nnzb);
    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        b_rowidx[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
            nnzb[blk_idx] * sizeof(unsigned int), 64);
        b_values[blk_idx] =
            (float *)libxsmm_aligned_malloc(nnzb[blk_idx] * sizeof(float), 64);
    }
    BlockSpMatStep2(K, C, KB, CB, colptr, rowidx, values, b_colptr, b_rowidx,
                    b_values);
    /* touch C */
    for (l_n = 0; l_n < N / NB; ++l_n) {
        for (l_k = 0; l_k < K / KB; ++l_k) {
            for (l_nn = 0; l_nn < NB / nb; ++l_nn) {
                for (l_kk = 0; l_kk < KB; ++l_kk) {
                    for (l_nnn = 0; l_nnn < nb; ++l_nnn) {
                        LIBXSMM_VLA_ACCESS(5, l_p_C_gold, l_n, l_k, l_nn, l_kk,
                                           l_nnn, K / KB, NB / nb, KB, nb) =
                            0.0f;
                        LIBXSMM_VLA_ACCESS(5, l_p_C, l_n, l_k, l_nn, l_kk,
                                           l_nnn, K / KB, NB / nb, KB, nb) =
                            0.0f;
                    }
                }
            }
        }
    }
    /* dense routine */
    for (l_n = 0; l_n < N / NB; ++l_n) {
        for (l_k = 0; l_k < K / KB; ++l_k) {
            for (l_c = 0; l_c < C / CB; ++l_c) {
                for (l_nn = 0; l_nn < NB / nb; ++l_nn) {
                    for (l_kk = 0; l_kk < KB; ++l_kk) {
                        k = l_k * KB + l_kk;
                        for (l_cc = 0; l_cc < CB; ++l_cc) {
                            c = l_c * CB + l_cc;
                            for (l_nnn = 0; l_nnn < nb; ++l_nnn) {
                                LIBXSMM_VLA_ACCESS(5, l_p_C_gold, l_n, l_k,
                                                   l_nn, l_kk, l_nnn, K / KB,
                                                   NB / nb, KB, nb) +=
                                    LIBXSMM_VLA_ACCESS(5, l_p_A, l_n, l_c, l_nn,
                                                       l_cc, l_nnn, C / CB,
                                                       NB / nb, CB, nb) *
                                    l_B[k * C + c];
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
            num_blocks * sizeof(libxsmm_gemm_descriptor *), 64);
    libxsmm_smmfunction *mykernel =
        (libxsmm_smmfunction *)libxsmm_aligned_malloc(
            num_blocks * sizeof(libxsmm_smmfunction), 64);
    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        l_xgemm_desc[blk_idx] = libxsmm_gemm_descriptor_dinit(
            &l_xgemm_blob, LIBXSMM_GEMM_PRECISION(float), NB / nb, KB, CB, CB,
            0, KB, alpha, beta, flags, prefetch);
        mykernel[blk_idx] =
            libxsmm_create_xcsc_soa(l_xgemm_desc[blk_idx], b_colptr[blk_idx],
                                    b_rowidx[blk_idx],
                                    (const void *)b_values[blk_idx], nb).smm;
    }
#ifdef _OPENMP
#   pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(k,n,c)
#endif
    for (k = 0; k < K / KB; ++k) {
        for (n = 0; n < N / NB; ++n) {
            for (c = 0; c < C / CB; ++c) {
                mykernel[k * C / CB + c](&(l_A[(n * C / CB + c) * CB * NB]),
                                         b_values[k * C / CB + c],
                                         &(l_C[(n * K / KB + k) * NB * KB]));
            }
        }
    }
    /* check error */
    float l_max_error = 0.0f;
    for (i = 0; i < N * K; ++i) {
        if (fabs(l_C[i] - l_C_gold[i]) > l_max_error) {
            l_max_error = (float)fabs(l_C[i] - l_C_gold[i]);
        }
    }
    printf("max error = %f\n", l_max_error);
    /* check performace */
    unsigned long long l_start = libxsmm_timer_tick();
    for (i = 0; i < (int)REPS; ++i) {
#ifdef _OPENMP
#       pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(k,n,c)
#endif
        for (k = 0; k < K / KB; ++k) {
            for (n = 0; n < N / NB; ++n) {
                for (c = 0; c < C / CB; ++c) {
                    mykernel[k * C / CB + c](
                        &(l_A[(n * C / CB + c) * CB * NB]),
                        b_values[k * C / CB + c],
                        &(l_C[(n * K / KB + k) * NB * KB]));
                }
            }
        }
    }
    unsigned long long l_end = libxsmm_timer_tick();
    double l_total = libxsmm_timer_duration(l_start, l_end);
    printf("%fs for sparse (asm)\n", l_total);
    printf("%f GFLOPS for sparse (asm)\n",
           ((double)((double)REPS * (double)N * (double)C * (double)K) * 2.0) /
               (l_total * 1.0e9));
    /* clean up */
    libxsmm_free(l_A);
    libxsmm_free(l_B);
    libxsmm_free(l_C);
    libxsmm_free(l_C_gold);
    for (blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        libxsmm_free(b_values[blk_idx]);
        libxsmm_free(b_colptr[blk_idx]);
        libxsmm_free(b_rowidx[blk_idx]);
    }
    libxsmm_free(b_values);
    libxsmm_free(b_colptr);
    libxsmm_free(b_rowidx);

    return 0;
}
