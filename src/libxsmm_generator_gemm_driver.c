/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include <libxsmm_generator.h>
#include <libxsmm_macros.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>


LIBXSMM_INLINE void print_help(void) {
  printf("\nwrong usage -> exit!\n\n\n");
  printf("Usage (sparse*dense=dense, dense*sparse=dense):\n");
  printf("    sparse, sparse_csr, sparse_csr_reg, sparse_csr_soa\n");
  printf("    filename to append\n");
  printf("    routine name\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA (if < 1 --> A sparse)\n");
  printf("    LDB (if < 1 --> B sparse)\n");
  printf("    LDC\n");
  printf("    alpha: -1 or 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned (ignored for sparse)\n");
  printf("    0: unaligned C, otherwise aligned (ignored for sparse)\n");
  printf("    ARCH: noarch, wsm, snb, hsw, knc, knl, knm, skx\n");
  printf("    PREFETCH: nopf (none), pfsigonly, other dense options fall-back to pfsigonly\n");
  printf("    PRECISION: SP, DP\n");
  printf("    matrix input (CSC mtx file)\n");
  printf("\n\n");
  printf("Usage (dense*dense=dense):\n");
  printf("    dense, dense_asm\n");
  printf("    filename to append\n");
  printf("    routine name\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA\n");
  printf("    LDB\n");
  printf("    LDC\n");
  printf("    alpha: -1 or 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    ARCH: noarch, wsm, snb, hsw, knc, knl, knm, skx\n");
  printf("    PREFETCH: nopf (none), pfsigonly, BL2viaC, AL2, curAL2,\n"
         "              AL2jpst, AL2_BL2viaC, curAL2_BL2viaC,\n"
         "              AL2jpst_BL2viaC, AL1, BL1, CL1,\n"
         "              AL1_BL1, BL1_CL1, AL1_CL1, AL1_BL1_CL1\n");
  printf("    PRECISION: SP, DP\n");
  printf("\n\n\n\n");
}

int main(int argc, char* argv []) {
  libxsmm_gemm_descriptor l_xgemm_desc;
  char* l_type;
  char* l_file_out;
  char* l_matrix_file_in;
  char* l_routine_name;
  char* l_arch;
  char* l_precision;
  int l_prefetch;
  int l_m = 0;
  int l_n = 0;
  int l_k = 0;
  int l_lda = 0;
  int l_ldb = 0;
  int l_ldc = 0;
  int l_aligned_a = 0;
  int l_aligned_c = 0;
  double l_alpha = 0;
  double l_beta = 0;
  int l_single_precision = 0;
  int l_is_csr = 0;

  /* check argument count for a valid range */
  if (argc != 17 && argc != 18) {
    print_help();
    return -1;
  }

  /* names of files and routines */
  l_type = argv[1];
  l_file_out = argv[2];
  l_routine_name = argv[3];

  /* xgemm sizes */
  l_m = atoi(argv[4]);
  l_n = atoi(argv[5]);
  l_k = atoi(argv[6]);
  l_lda = atoi(argv[7]);
  l_ldb = atoi(argv[8]);
  l_ldc = atoi(argv[9]);

  /* condense < 1 to 0 for lda and ldb */
  if ( l_lda < 1 )
    l_lda = 0;
  if ( l_ldb < 1 )
    l_ldb = 0;

  /* some sugar */
  l_alpha = atof(argv[10]);
  l_beta = atof(argv[11]);
  l_aligned_a = atoi(argv[12]);
  l_aligned_c = atoi(argv[13]);

  /* arch specific stuff */
  l_arch = argv[14];
  l_precision = argv[16];

  /* some intial parameters checks */
  /* check for sparse / dense only */
  if ( (strcmp(l_type, "sparse")         != 0) &&
       (strcmp(l_type, "sparse_csr")     != 0) &&
       (strcmp(l_type, "sparse_csr_reg") != 0) &&
       (strcmp(l_type, "sparse_csr_soa") != 0) &&
       (strcmp(l_type, "dense")          != 0) &&
       (strcmp(l_type, "dense_asm")      != 0) ) {
    print_help();
    return -1;
  }

  /* check for the right number of arguments depending on type */
  if ( ( (strcmp(l_type, "sparse") == 0) && (argc != 18) )         ||
       ( (strcmp(l_type, "sparse_csr") == 0) && (argc != 18) )     ||
       ( (strcmp(l_type, "sparse_csr_reg") == 0) && (argc != 18) ) ||
       ( (strcmp(l_type, "sparse_csr_soa") == 0) && (argc != 18) ) ||
       ( (strcmp(l_type, "dense")  == 0) && (argc != 17) )         ||
       ( (strcmp(l_type, "dense_asm")  == 0) && (argc != 17) ) ) {
    print_help();
    return -1;
  }

  /* set value of prefetch flag */
  if (strcmp("nopf", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_NONE;
  }
  else if (strcmp("pfsigonly", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_SIGONLY;
  }
  else if (strcmp("BL2viaC", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_BL2_VIA_C;
  }
  else if (strcmp("curAL2", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2_AHEAD;
  }
  else if (strcmp("curAL2_BL2viaC", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD;
  }
  else if (strcmp("AL2", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2;
  }
  else if (strcmp("AL2_BL2viaC", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C;
  }
  else if (strcmp("AL2jpst", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2_JPST;
  }
  else if (strcmp("AL2jpst_BL2viaC", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST;
  }
  else if (strcmp("AL1", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL1;
  }
  else if (strcmp("BL1", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_BL1;
  }
  else if (strcmp("CL1", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_CL1;
  }
  else if (strcmp("AL1_BL1", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL1_BL1;
  }
  else if (strcmp("BL1_CL1", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_BL1_CL1;
  }
  else if (strcmp("AL1_CL1", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL1_CL1;
  }
  else if (strcmp("AL1_BL1_CL1", argv[15]) == 0) {
    l_prefetch = LIBXSMM_PREFETCH_AL1_BL1_CL1;
  }
  else {
    print_help();
    return -1;
  }

  /* check value of arch flag */
  if ( (strcmp(l_arch, "wsm") != 0)    &&
       (strcmp(l_arch, "snb") != 0)    &&
       (strcmp(l_arch, "hsw") != 0)    &&
       (strcmp(l_arch, "knc") != 0)    &&
       (strcmp(l_arch, "knl") != 0)    &&
       (strcmp(l_arch, "knm") != 0)    &&
       (strcmp(l_arch, "skx") != 0)    &&
       (strcmp(l_arch, "noarch") != 0) ) {
    print_help();
    return -1;
  }

  /* check and evaluate precision flag */
  if ( strcmp(l_precision, "SP") == 0 ) {
    l_single_precision = 1;
  } else if ( strcmp(l_precision, "DP") == 0 ) {
    l_single_precision = 0;
  } else {
    print_help();
    return -1;
  }

  /* check alpha */
  if ((l_alpha < -1 || -1 < l_alpha) && (l_alpha < 1 || 1 < l_alpha)) {
    print_help();
    return -1;
  }

  /* check beta */
  if ((l_beta < 0 || 0 < l_beta) && (l_beta < 1 || 1 < l_beta)) {
    print_help();
    return -1;
  }

  LIBXSMM_GEMM_DESCRIPTOR(l_xgemm_desc, 0 == l_single_precision ? LIBXSMM_GEMM_PRECISION_F64 : LIBXSMM_GEMM_PRECISION_F32,
    (0 != l_aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0) | (0 != l_aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0),
    l_m, l_n, l_k, l_lda, l_ldb, l_ldc,
    l_alpha, l_beta, l_prefetch);

  if ( strcmp(l_type, "sparse") == 0 || strcmp(l_type, "sparse_csr") == 0 ||
       strcmp(l_type, "sparse_csr_reg") == 0 || strcmp(l_type, "sparse_csr_soa") == 0 ) {
    /* read additional paramter for CSC/CSR description */
    l_matrix_file_in = argv[17];

    /* some more restrictive checks are needed in case of sparse */
    if ( (l_alpha < 1) || (1 < l_alpha) ) {
      print_help();
      return -1;
    }

    if (l_lda < 1 && l_ldb < 1) {
      print_help();
      return -1;
    }

    if (l_ldc < 1) {
      print_help();
      return -1;
    }

    if ( strcmp(l_type, "sparse_csr") == 0 ) {
      l_is_csr = 1;
    }
    if ( strcmp(l_type, "sparse_csr_soa") == 0 ) {
      l_is_csr = 2;
    }
    if ( strcmp(l_type, "sparse_csr_reg") == 0 ) {
      l_is_csr = 3;
    }

    libxsmm_generator_spgemm( l_file_out, l_routine_name, &l_xgemm_desc, l_arch, l_matrix_file_in, l_is_csr );
  }

  if ( (strcmp(l_type, "dense")     == 0) ||
       (strcmp(l_type, "dense_asm") == 0) ) {
    if (l_lda < 1 || l_ldb < 1 || l_ldc < 1) {
      print_help();
      return -1;
    }

    if ( strcmp(l_type, "dense")  == 0 ) {
      libxsmm_generator_gemm_inlineasm( l_file_out, l_routine_name, &l_xgemm_desc, l_arch );
    } else {
      libxsmm_generator_gemm_directasm( l_file_out, l_routine_name, &l_xgemm_desc, l_arch );
    }
  }

  return 0;
}

