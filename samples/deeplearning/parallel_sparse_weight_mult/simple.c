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
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void BlockSpMatStep1(int K, int C, int KB, int CB, unsigned int *colptr,
                     unsigned int *rowidx, unsigned int *b_colptr[],
                     int *nnzb) {
    int num_blocks = K / KB * C / CB;
    for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        nnzb[blk_idx] = 0;
        for (int i = 0; i <= KB; ++i) {
            b_colptr[blk_idx][i] = 0;
        }
    }
    for (int k = 0; k < K; ++k) {
        int k_blk_idx = k / KB;
        int k_blk_offset = k % KB;
        unsigned colstart = colptr[k];
        unsigned colend = colptr[k + 1];
        for (int i = colstart; i < colend; ++i) {
            int c = rowidx[i];
            int c_blk_idx = c / CB;
            int blk_idx = k_blk_idx * C / CB + c_blk_idx;
            nnzb[blk_idx]++;
            b_colptr[blk_idx][k_blk_offset + 1]++;
        }
    }
    for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        for (int i = 0; i < KB; ++i) {
            b_colptr[blk_idx][i + 1] += b_colptr[blk_idx][i];
        }
    }
}

void BlockSpMatStep2(int K, int C, int KB, int CB, unsigned int *colptr,
                     unsigned int *rowidx, float *values,
                     unsigned int *b_colptr[], unsigned int *b_rowidx[],
                     float *b_values[]) {
    int num_blocks = K / KB * C / CB;
    for (int k = 0; k < K; ++k) {
        int k_blk_idx = k / KB;
        int k_blk_offset = k % KB;
        unsigned colstart = colptr[k];
        unsigned colend = colptr[k + 1];
        for (int i = colstart; i < colend; ++i) {
            int c = rowidx[i];
            int c_blk_idx = c / CB;
            int c_blk_offset = c % CB;
            int blk_idx = k_blk_idx * C / CB + c_blk_idx;
            b_rowidx[blk_idx][b_colptr[blk_idx][k_blk_offset]] = c_blk_offset;
            b_values[blk_idx][b_colptr[blk_idx][k_blk_offset]] = values[i];
            b_colptr[blk_idx][k_blk_offset]++;
        }
    }

    for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        for (int i = KB; i > 0; --i) {
            b_colptr[blk_idx][i] = b_colptr[blk_idx][i - 1];
        }
        b_colptr[blk_idx][0] = 0;
    }
}

int main(int argc, char **argv) {
    int N = (argc == 6) ? atoi(argv[1]) : 2048;
    int C = (argc == 6) ? atoi(argv[2]) : 512;
    int K = (argc == 6) ? atoi(argv[3]) : 512;
    int NB = (argc == 6) ? atoi(argv[4]) : 32;
    int CB = (argc == 6) ? atoi(argv[5]) : 128;
    int KB = (argc == 6) ? atoi(argv[6]) : 128;
    unsigned int SPAR = (argc == 6) ? atoi(argv[7]) : 90;
    unsigned int REPS = (argc == 6) ? atoi(argv[8]) : 10;
    assert(K % KB == 0);
    assert(C % CB == 0);
    double sparse_frac = ((double)SPAR / (double)100.0);
    int nb = 16;

    libxsmm_gemm_prefetch_type prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
    int flags = LIBXSMM_GEMM_FLAGS('N', 'N');
    float *l_a = (float *)libxsmm_aligned_malloc(sizeof(float) * N * C, 64);
    float *l_b = (float *)libxsmm_aligned_malloc(sizeof(float) * C * K, 64);
    float *l_c = (float *)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
    float *l_c_gold =
        (float *)libxsmm_aligned_malloc(sizeof(float) * N * K, 64);
    LIBXSMM_VLA_DECL(5, float, l_p_a, l_a, C / CB, NB / nb, CB, nb);
    LIBXSMM_VLA_DECL(5, float, l_p_c, l_c, K / KB, NB / nb, KB, nb);
    LIBXSMM_VLA_DECL(5, float, l_p_c_gold, l_c_gold, K / KB, NB / nb, KB, 16);
    /* touch A */
    for (int l_n = 0; l_n < N / NB; ++l_n) {
        for (int l_c = 0; l_c < C / CB; ++l_c) {
            for (int l_nn = 0; l_nn < NB / nb; ++l_nn) {
                for (int l_cc = 0; l_cc < CB; ++l_cc) {
                    for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                        LIBXSMM_VLA_ACCESS(5, l_p_a, l_n, l_c, l_nn, l_cc,
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
    for (int l_k = 0; l_k < K; l_k++) {
        colptr[l_k + 1] = 0;
        for (int l_c = 0; l_c < C; l_c++) {
            double tmp = libxsmm_rng_f64();
            if (tmp < sparse_frac) {
                tmp = 0.0;
            } else {
                nnz++;
                colptr[l_k + 1]++;
            }
            l_b[l_k * C + l_c] = tmp;
        }
    }
    for (int l_k = 0; l_k < K; l_k++) {
        colptr[l_k + 1] += colptr[l_k];
    }
    unsigned int *rowidx =
        (unsigned int *)libxsmm_aligned_malloc(nnz * sizeof(unsigned int), 64);
    float *values = (float *)libxsmm_aligned_malloc(nnz * sizeof(float), 64);
    for (int l_k = 0; l_k < K; l_k++) {
        int offset = colptr[l_k];
        for (int l_c = 0; l_c < C; l_c++) {
            if (l_b[l_k * C + l_c] != 0) {
                rowidx[offset] = l_c;
                values[offset] = l_b[l_k * C + l_c];
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
    for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        b_colptr[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
            (KB + 1) * sizeof(unsigned int), 64);
    }
    BlockSpMatStep1(K, C, KB, CB, colptr, rowidx, b_colptr, nnzb);
    for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        b_rowidx[blk_idx] = (unsigned int *)libxsmm_aligned_malloc(
            nnzb[blk_idx] * sizeof(unsigned int), 64);
        b_values[blk_idx] =
            (float *)libxsmm_aligned_malloc(nnzb[blk_idx] * sizeof(float), 64);
    }
    BlockSpMatStep2(K, C, KB, CB, colptr, rowidx, values, b_colptr, b_rowidx,
                    b_values);
    /* touch C */
    for (int l_n = 0; l_n < N / NB; ++l_n) {
        for (int l_k = 0; l_k < K / KB; ++l_k) {
            for (int l_nn = 0; l_nn < NB / nb; ++l_nn) {
                for (int l_kk = 0; l_kk < KB; ++l_kk) {
                    for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                        LIBXSMM_VLA_ACCESS(5, l_p_c_gold, l_n, l_k, l_nn, l_kk,
                                           l_nnn, K / KB, NB / nb, KB, nb) =
                            0.0f;
                        LIBXSMM_VLA_ACCESS(5, l_p_c, l_n, l_k, l_nn, l_kk,
                                           l_nnn, K / KB, NB / nb, KB, nb) =
                            0.0f;
                    }
                }
            }
        }
    }
    /* dense routine */
    for (int l_n = 0; l_n < N / NB; ++l_n) {
        for (int l_k = 0; l_k < K / KB; ++l_k) {
            for (int l_c = 0; l_c < C / CB; ++l_c) {
                for (int l_nn = 0; l_nn < NB / nb; ++l_nn) {
                    for (int l_kk = 0; l_kk < KB; ++l_kk) {
                        int k = l_k * KB + l_kk;
                        for (int l_cc = 0; l_cc < CB; ++l_cc) {
                            int c = l_c * CB + l_cc;
                            for (int l_nnn = 0; l_nnn < nb; ++l_nnn) {
                                LIBXSMM_VLA_ACCESS(5, l_p_c_gold, l_n, l_k,
                                                   l_nn, l_kk, l_nnn, K / KB,
                                                   NB / nb, KB, nb) +=
                                    LIBXSMM_VLA_ACCESS(5, l_p_a, l_n, l_c, l_nn,
                                                       l_cc, l_nnn, C / CB,
                                                       NB / nb, CB, nb) *
                                    l_b[k * C + c];
                            }
                        }
                    }
                }
            }
        }
    }
    // FWD
    float alpha = 1.0;
    float beta = 1.0;
    libxsmm_descriptor_blob l_xgemm_blob;
    libxsmm_gemm_descriptor **l_xgemm_desc =
        (libxsmm_gemm_descriptor **)libxsmm_aligned_malloc(
            num_blocks * sizeof(libxsmm_gemm_descriptor *), 64);
    libxsmm_smmfunction *mykernel =
        (libxsmm_smmfunction *)libxsmm_aligned_malloc(
            num_blocks * sizeof(libxsmm_smmfunction), 64);
    for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        l_xgemm_desc[blk_idx] = libxsmm_gemm_descriptor_dinit(
            &l_xgemm_blob, LIBXSMM_GEMM_PRECISION(float), NB / nb, KB, CB, CB,
            0, KB, alpha, beta, flags, prefetch);
        mykernel[blk_idx] =
            libxsmm_create_xcsc_soa(l_xgemm_desc[blk_idx], b_colptr[blk_idx],
                                    b_rowidx[blk_idx],
                                    (const void *)b_values[blk_idx])
                .smm;
    }
#pragma omp parallel for collapse(2)
    for (int k = 0; k < K / KB; ++k) {
        for (int n = 0; n < N / NB; ++n) {
            for (int c = 0; c < C / CB; ++c) {
                mykernel[k * C / CB + c](&(l_a[(n * C / CB + c) * CB * NB]),
                                         b_values[k * C / CB + c],
                                         &(l_c[(n * K / KB + k) * NB * KB]));
            }
        }
    }
    // check error
    float l_max_error = 0.0f;
    for (int i = 0; i < N * K; ++i) {
        if (fabs(l_c[i] - l_c_gold[i]) > l_max_error) {
            l_max_error = (float)fabs(l_c[i] - l_c_gold[i]);
        }
    }
    printf("max error = %f\n", l_max_error);
    // check performace
    unsigned long long l_start = libxsmm_timer_tick();
    for (int i = 0; i < REPS; ++i) {
#pragma omp parallel for collapse(2)
        for (int k = 0; k < K / KB; ++k) {
            for (int n = 0; n < N / NB; ++n) {
                for (int c = 0; c < C / CB; ++c) {
                    mykernel[k * C / CB + c](
                        &(l_a[(n * C / CB + c) * CB * NB]),
                        b_values[k * C / CB + c],
                        &(l_c[(n * K / KB + k) * NB * KB]));
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
    // clean up
    libxsmm_free(l_a);
    libxsmm_free(l_c);
    for (int blk_idx = 0; blk_idx < num_blocks; ++blk_idx) {
        libxsmm_free(b_values[blk_idx]);
        libxsmm_free(b_colptr[blk_idx]);
        libxsmm_free(b_rowidx[blk_idx]);
    }
    libxsmm_free(b_values);
    libxsmm_free(b_colptr);
    libxsmm_free(b_rowidx);

    return 0;
}