/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#include "generator_common.h"
#include "generator_dense_common.h"
#include "generator_dense_instructions.h"
/*
#include "generator_dense_imci_avx512_common.h"
#include "generator_dense_imci_microkernel.h"
#include "generator_dense_avx512_microkernel.h"
*/

int libxsmm_generator_dense_imci_avx512_kernel_k_loop( libxsmm_generated_code*            io_generated_code,
                                                       const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                       const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                       const libxsmm_xgemm_descriptor*    i_xgemm_desc,
                                                       const char*                        i_arch,
                                                       unsigned int                       i_n_blocking ) {
  const unsigned int k_blocking = 8;
  const unsigned int k_threshold = 8;
  unsigned int KisFullyUnrolled = 0;

#if 0
  /* Let's do something special for SeisSol high-order (N == 9 holds true) */
  /*if ((K != 9) && (K >= 8) && (N == 9)) {
    avx512_kernel_8x9xKfullyunrolled_indexed_dp_asm(codestream, K, lda, ldb, ldc, alignA, tPrefetch, bUseMasking);
    KisFullyUnrolled = true;
  } else if ((K == 9) && (N == 9)) {
    avx512_kernel_8x9x9fullyunrolled_indexed_dp_asm(codestream, K, lda, ldb, ldc, alignA, tPrefetch, bUseMasking);
    KisFullyUnrolled = true;
  } else*/ if (K % k_blocking == 0 && K >= k_threshold) {
    avx512_header_kloop_dp_asm(codestream, 8, k_blocking);
    if (k_blocking == 8) {
      avx512_kernel_8xNx8_dp_asm(codestream, N, lda, ldb, ldc, alignA, tPrefetch, bUseMasking);
    } else {
      std::cout << " !!! ERROR, AVX-512, k-blocking !!! " << std::endl;
      exit(-1);
    }
    avx512_footer_kloop_dp_asm(codestream, 8, K);
  } else {
    int max_blocked_K = (K/k_blocking)*k_blocking;
    if (max_blocked_K > 0 ) {
      avx512_header_kloop_dp_asm(codestream, 8, k_blocking);
      avx512_kernel_8xNx8_dp_asm(codestream, N, lda, ldb, ldc, alignA, tPrefetch, bUseMasking);
      avx512_footer_kloop_notdone_dp_asm(codestream, 8, max_blocked_K);
    }
    for (int i = max_blocked_K; i < K; i++) {
      avx512_kernel_8xN_dp_asm(codestream, N, lda, ldb, ldc, alignA, i-max_blocked_K, tPrefetch, bUseMasking);
    }
    // update r8 and r9
    codestream << "                         \"addq $" << lda * 8 * (K - max_blocked_K) << ", %%r9\\n\\t\"" << std::endl;
    // next A prefetch "same" rows in "same" column, but in a different matrix 
    if (    (tPrefetch.compare("AL2jpst") == 0) 
         || (tPrefetch.compare("AL2jpst_BL2viaC") == 0)
         || (tPrefetch.compare("AL2") == 0)
         || (tPrefetch.compare("AL2_BL2viaC") == 0)) {
      codestream << "                         \"addq $" << lda * 8 * (K - max_blocked_K) << ", %%r11\\n\\t\"" << std::endl;
    }
    if (max_blocked_K > 0 ) {
      codestream << "                         \"subq $" << max_blocked_K * 8 << ", %%r8\\n\\t\"" << std::endl;
    }
  }
#endif
  return KisFullyUnrolled;
}

void libxsmm_generator_dense_imci_avx512_kernel_m_loop( libxsmm_generated_code*            io_generated_code,
                                                        const libxsmm_gp_reg_mapping*      i_gp_reg_mapping,
                                                        const libxsmm_micro_kernel_config* i_micro_kernel_config,
                                                        const libxsmm_xgemm_descriptor*    i_xgemm_desc,
                                                        const char*                        i_arch,
                                                        unsigned int                       i_n_blocking ) {
  /* we proceed as much as we can in vector length steps, remainder is handled uisng masking */
  int l_m_done = (i_xgemm_desc->m / i_micro_kernel_config->vector_length) * i_micro_kernel_config->vector_length;

  /* multiples of vector_length in M */
  if (l_m_done > 0) {
    libxsmm_generator_dense_header_mloop( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_micro_kernel_config->vector_length );
    libxsmm_generator_dense_load_C( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, 
                                    i_xgemm_desc, i_micro_kernel_config->vector_length, i_n_blocking );

    /*bool KisFullyUnrolled = avx512_generate_inner_k_loop_dp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, tPrefetch, false);*/

    libxsmm_generator_dense_store_C( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, 
                                     i_xgemm_desc, i_micro_kernel_config->vector_length, i_n_blocking  );
    libxsmm_generator_dense_footer_mloop( io_generated_code, i_gp_reg_mapping, i_micro_kernel_config, i_xgemm_desc, 
                                          i_micro_kernel_config->vector_length, l_m_done );
  }

  /* Remainder Handling using Masking, we are using M loop counter register as GP register for the mask */
  if ( l_m_done != i_xgemm_desc->m ) {
    /* request masking support, @TODO performance penality here, as a new object is created */
    libxsmm_micro_kernel_config l_micro_kernel_config_mask;
    libxsmm_generator_dense_init_micro_kernel_config_fullvector( &l_micro_kernel_config_mask, i_xgemm_desc, i_arch, 1 );

    /* initialize k1 register */

    /* run masked micro kernel */    

#if 0
    switch(M - mDone)
    {
      case 1: 
        codestream << "                         \"movq $1, %%r14\\n\\t\"" << std::endl;
        break;
      case 2: 
        codestream << "                         \"movq $3, %%r14\\n\\t\"" << std::endl;
        break;
      case 3: 
        codestream << "                         \"movq $7, %%r14\\n\\t\"" << std::endl;
        break;
      case 4: 
        codestream << "                         \"movq $15, %%r14\\n\\t\"" << std::endl;
        break;
      case 5: 
        codestream << "                         \"movq $31, %%r14\\n\\t\"" << std::endl;
        break;
      case 6: 
        codestream << "                         \"movq $63, %%r14\\n\\t\"" << std::endl;
        break;
      case 7: 
        codestream << "                         \"movq $127, %%r14\\n\\t\"" << std::endl;
        break;       
    }
    codestream << "                         \"kmovw %%r14d, %%k1\\n\\t\"" << std::endl;
    avx512_load_8xN_dp_asm(codestream, N, ldc, alignC, bAdd, true);
    // innner loop over K
    avx512_generate_inner_k_loop_dp(codestream, lda, ldb, ldc, M, N, K, alignA, alignC, bAdd, tPrefetch, true);
    avx512_store_8xN_dp_asm(codestream, N, ldc, alignC, tPrefetch, true);
    codestream << "                         \"addq $" << (M - mDone) * 8 << ", %%r10\\n\\t\"" << std::endl;
    codestream << "                         \"subq $" << (K * 8 * lda) - ( (M - mDone) * 8) << ", %%r9\\n\\t\"" << std::endl;
#else
    fprintf(stderr, "ERROR MASKING AVX512\n");
    exit(-1);
#endif
  }
}

void libxsmm_generator_dense_imci_avx512_kernel( libxsmm_generated_code*         io_generated_code,
                                                 const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                                 const char*                     i_arch ) {
  /* define gp register mapping */
  libxsmm_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* machting calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_R15; /* masking */
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_RAX; /* B stride helper */
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_RBX; /* B stride helper */
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_R9;  /* B stride helper */
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_R10; /* B stride helper */
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_R11; /* B stride helper */

  /* define the micro kernel code gen properties */
  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_generator_dense_init_micro_kernel_config_fullvector( &l_micro_kernel_config, i_xgemm_desc, i_arch, 0 );

  /* set up architecture dependent compute micro kernel generator */
  void (*l_generator_microkernel)(libxsmm_generated_code*, const libxsmm_gp_reg_mapping*, const libxsmm_micro_kernel_config*, 
                                  const libxsmm_xgemm_descriptor*, const unsigned int, const unsigned int, const int);
  if ( (strcmp(i_arch, "knc") == 0) ) {
    /*l_generator_microkernel = libxsmm_generator_dense_imci_microkernel;*/
  } else if ( (strcmp(i_arch, "knl") == 0) ) {
    /*l_generator_microkernel = libxsmm_generator_dense_avx512_microkernel;*/
  } else if ( (strcmp(i_arch, "skx") == 0) ) {
    /*l_generator_microkernel = libxsmm_generator_dense_avx512_microkernel;*/
  } else {
    fprintf(stderr, "LIBXSMM ERROR libxsmm_generator_dense_imci_avx512_kernel, cannot select microkernel\n");
    exit(-1);
  }

  unsigned int l_number_of_chunks = 1+((i_xgemm_desc->n-1)/30);
  unsigned int l_modulo = i_xgemm_desc->n%l_number_of_chunks;
  unsigned int l_n2 = i_xgemm_desc->n/l_number_of_chunks;
  unsigned int l_n1 = l_n2 + 1;
  unsigned int l_N2 = 0;
  unsigned int l_N1 = 0;
  unsigned int l_chunk = 0;
  if (l_n1 > 30) l_n1 = 30; /* this just the case if i_xgemm_desc->n/l_number_of_chunks has no remainder */
  for (l_chunk = 0; l_chunk < l_number_of_chunks; l_chunk++) {
    if (l_chunk < l_modulo) {
      l_N1 += l_n1;
    } else {
      l_N2 += l_n2;
    }
  }
  
  printf("N splitting of DP AVX512 Kernel: %i %i %i %i\n", l_N1, l_N2, l_n1, l_n2);

  /* open asm */
  libxsmm_generator_dense_x86_open_instruction_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );

  if (l_number_of_chunks == 1) {
    libxsmm_generator_dense_imci_avx512_kernel_m_loop( io_generated_code, &l_gp_reg_mapping, &l_micro_kernel_config,
                                                       i_xgemm_desc, i_arch, l_n2); 
  } else {
#if 0
    if ((l_N2 > 0) && (l_N1 > 0)) {
      avx512_header_nloop_dp_asm(codestream, l_n1);
      avx512_generate_inner_m_loop_dp(codestream, lda, ldb, ldc, M, l_n1, K, alignA, alignC, bAdd, tPrefetch);
      avx512_footer_nloop_dp_asm(codestream, l_n1, l_N1, M, lda, ldb, ldc);

      avx512_header_nloop_dp_asm(codestream, l_n2);
      avx512_generate_inner_m_loop_dp(codestream, lda, ldb, ldc, M, l_n2, K, alignA, alignC, bAdd, tPrefetch);
      avx512_footer_nloop_dp_asm(codestream, l_n2, N, M, lda, ldb, ldc);
    } else if ((l_N2 > 0) && (l_N1 == 0)) {
      avx512_header_nloop_dp_asm(codestream, l_n2);
      avx512_generate_inner_m_loop_dp(codestream, lda, ldb, ldc, M, l_n2, K, alignA, alignC, bAdd, tPrefetch);
      avx512_footer_nloop_dp_asm(codestream, l_n2, N, M, lda, ldb, ldc);
    } else {
      std::cout << " !!! ERROR, AVX512 n-blocking !!! " << std::endl;
      exit(-1);
    }
#endif
  }
  
  /* close asm */
  libxsmm_generator_dense_x86_close_instruction_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );
}

