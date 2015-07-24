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

#include "generator_common.h"

void libxsmm_generator_dense_add_isa_check_header( char**       io_generated_code, 
                                                   const char*  i_arch ) {
  if ( (strcmp( i_arch, "wsm" ) == 0) ) {
    libxsmm_append_string( io_generated_code, "#ifdef __SSE3__\n");
    libxsmm_append_string( io_generated_code, "#ifdef __AVX__\n");
    libxsmm_append_string( io_generated_code, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling SSE3 code on AVX or newer architecture: \" __FILE__)\n");
    libxsmm_append_string( io_generated_code, "#endif\n");
  } else if ( (strcmp( i_arch, "snb" ) == 0) ) {
    libxsmm_append_string( io_generated_code, "#ifdef __AVX__\n");
    libxsmm_append_string( io_generated_code, "#ifdef __AVX2__\n");
    libxsmm_append_string( io_generated_code, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling AVX code on AVX2 or newer architecture: \" __FILE__)\n");
    libxsmm_append_string( io_generated_code, "#endif\n");
  } else if ( (strcmp( i_arch, "hsw" ) == 0) ) {
    libxsmm_append_string( io_generated_code, "#ifdef __AVX2__\n");
    libxsmm_append_string( io_generated_code, "#ifdef __AVX512F__\n");
    libxsmm_append_string( io_generated_code, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling AVX2 code on AVX512 or newer architecture: \" __FILE__)\n");
    libxsmm_append_string( io_generated_code, "#endif\n");
  } else if ( (strcmp( i_arch, "knl" ) == 0) ||
              (strcmp( i_arch, "skx" ) == 0) ) {
    libxsmm_append_string( io_generated_code, "#ifdef __AVX512F__\n");
  } else if ( (strcmp( i_arch, "noarch" ) == 0) ) {
    libxsmm_append_string( io_generated_code, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling arch-independent gemm kernel in: \" __FILE__)\n");
  } else {
    fprintf(stderr, "unsupported architecture in libxsmm_generator_dense_add_isa_check_header\n");
    exit(-1);   
  }
}

void libxsmm_generator_dense_add_isa_check_footer( char**       io_generated_code, 
                                                   const char*  i_arch ) {
  if ( (strcmp( i_arch, "wsm" ) == 0) || 
       (strcmp( i_arch, "snb" ) == 0) || 
       (strcmp( i_arch, "hsw" ) == 0) || 
       (strcmp( i_arch, "knc" ) == 0) || 
       (strcmp( i_arch, "knl" ) == 0) || 
       (strcmp( i_arch, "skx" ) == 0)    ) {
    libxsmm_append_string( io_generated_code, "#else\n");
    libxsmm_append_string( io_generated_code, "#pragma message (\"LIBXSMM KERNEL COMPILATION ERROR in: \" __FILE__)\n");
    libxsmm_append_string( io_generated_code, "#error No kernel was compiled, lacking support for current architecture?\n");
    libxsmm_append_string( io_generated_code, "#endif\n\n");
  } else if ( (strcmp( i_arch, "noarch" ) == 0) ) {
  } else {
    fprintf(stderr, "unsupported architecture in libxsmm_generator_dense_add_isa_check_footer\n");
    exit(-1);   
  }
}

void libxsmm_generator_dense_add_flop_counter( char**             io_generated_code, 
                                               const unsigned int i_m,
                                               const unsigned int i_n,
                                               const unsigned int i_k ) {
  char l_new_code[512];
  l_new_code[0] = '\0';
  
  libxsmm_append_string( io_generated_code, "#ifndef NDEBUG\n");
  libxsmm_append_string( io_generated_code, "#ifdef _OPENMP\n");
  libxsmm_append_string( io_generated_code, "#pragma omp atomic\n");
  libxsmm_append_string( io_generated_code, "#endif\n");
  sprintf( l_new_code, "libxsmm_num_total_flops += %i;\n", 2 * i_m * i_n * i_k);
  libxsmm_append_string( io_generated_code, l_new_code);
  libxsmm_append_string( io_generated_code, "#endif\n\n");
}

void libxsmm_generator_dense_sse_avx_open_kernel( char**      io_generated_code,
                                                  const char* i_prefetch) { 
  libxsmm_append_string( io_generated_code, "  __asm__ __volatile__(\"movq %0, %%r8\\n\\t\"\n" );
  libxsmm_append_string( io_generated_code, "                       \"movq %1, %%r9\\n\\t\"\n" );
  libxsmm_append_string( io_generated_code, "                       \"movq %2, %%r10\\n\\t\"\n" );
  if ( ( strcmp(i_prefetch, "BL2viaC") == 0 ) || 
       ( strcmp(i_prefetch, "BL1viaC") == 0 )    ) {
    libxsmm_append_string( io_generated_code, "                       \"movq %3, %%r12\\n\\t\"\n" );
  } else if ( ( strcmp(i_prefetch, "AL1viaC") == 0 ) ||
              ( strcmp(i_prefetch, "AL2") == 0 )        ) {
    libxsmm_append_string( io_generated_code, "                       \"movq %3, %%r11\\n\\t\"\n" );
  } else {}
  libxsmm_append_string( io_generated_code, "                       \"movq $0, %%r15\\n\\t\"\n" );
  libxsmm_append_string( io_generated_code, "                       \"movq $0, %%r14\\n\\t\"\n" );
  libxsmm_append_string( io_generated_code, "                       \"movq $0, %%r13\\n\\t\"\n" );  
  /* TODO Greg: how do we interface here? */
}

void libxsmm_generator_dense_sse_avx_close_kernel( char**      io_generated_code,
                                                   const char* i_prefetch) {
  if ( ( strcmp(i_prefetch, "BL2viaC") == 0 ) || 
       ( strcmp(i_prefetch, "BL1viaC") == 0 )    ) {
    libxsmm_append_string( io_generated_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(B_prefetch) : \"r8\",\"r9\",\"r10\",\"r12\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n" );
  } else if ( ( strcmp(i_prefetch, "AL1viaC") == 0 ) ||
       ( strcmp(i_prefetch, "AL2") == 0 )        ) {
    libxsmm_append_string( io_generated_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch) : \"r8\",\"r9\",\"r10\",\"r11\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n" );
  } else {
    libxsmm_append_string( io_generated_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C) : \"r8\",\"r9\",\"r10\",\"r13\",\"r14\",\"r15\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n" );
  }
  /* TODO Greg: how do we interface here? */
}

void libxsmm_generator_dense_header_kloop(char**             io_generated_code,
                                          const unsigned int i_gp_reg_kloop,
                                          const unsigned int i_m_blocking,
                                          const unsigned int i_k_blocking) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  libxsmm_instruction_alu_imm( io_generated_code, "movq", i_gp_reg_kloop, 0);
  sprintf( l_new_code, "2%i", i_m_blocking );
  libxsmm_instruction_register_jump_label( io_generated_code, l_new_code );
  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_kloop, i_k_blocking);
#if 0 
  codestream << "                         \"movq $0, %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"2" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << k_blocking << ", %%r13\\n\\t\"" << std::endl;
#endif
}

void libxsmm_generator_dense_footer_kloop(char**             io_generated_code,
                                          const unsigned int i_gp_reg_kloop,
                                          const unsigned int i_gp_reg_b,
                                          const unsigned int i_m_blocking,
                                          const unsigned int i_k,
                                          const unsigned int i_datatype_size,
                                          const unsigned int i_kloop_complete ) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  libxsmm_instruction_alu_imm( io_generated_code, "cmpq", i_gp_reg_kloop, i_k );
  sprintf( l_new_code, "2%ib", i_m_blocking );
  libxsmm_instruction_jump_to_label( io_generated_code, "jl", l_new_code );
  if ( i_kloop_complete !=0 ) {
    libxsmm_instruction_alu_imm( io_generated_code, "subq", i_gp_reg_b, i_k*i_datatype_size );
  }
#if 0
  codestream << "                         \"cmpq $" << K << ", %%r13\\n\\t\"" << std::endl;
  codestream << "                         \"jl 2" << m_blocking << "b\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << K * 8 << ", %%r8\\n\\t\"" << std::endl;
#endif
}

void libxsmm_generator_dense_header_nloop(char**             io_generated_code,
                                          const unsigned int i_gp_reg_mloop,
                                          const unsigned int i_gp_reg_nloop,
                                          const unsigned int i_n_blocking) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  sprintf( l_new_code, "1%i", i_n_blocking );
  libxsmm_instruction_register_jump_label( io_generated_code, l_new_code );
  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_nloop, i_n_blocking );
  libxsmm_instruction_alu_imm( io_generated_code, "movq", i_gp_reg_mloop, 0 );
#if 0
  codestream << "                         \"1" << n_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << n_blocking << ", %%r15\\n\\t\"" << std::endl;
  codestream << "                         \"movq $0, %%r14\\n\\t\"" << std::endl;
#endif
}


void libxsmm_generator_dense_footer_nloop(char**             io_generated_code,
                                          const unsigned int i_gp_reg_a,
                                          const unsigned int i_gp_reg_b,
                                          const unsigned int i_gp_reg_c,
                                          const unsigned int i_gp_reg_nloop,
                                          const unsigned int i_n_blocking,
                                          const unsigned int i_m,
                                          const unsigned int i_n,
                                          const unsigned int i_ldb,
                                          const unsigned int i_ldc,
                                          const char*        i_prefetch,
                                          const unsigned int i_gp_reg_pre_b,
                                          const unsigned int i_datatype_size) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_c, (i_n_blocking*i_ldc*i_datatype_size) - (i_m*i_datatype_size) );
  if ( strcmp( i_prefetch, "BL2viaC") == 0 ) {
    libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_pre_b, (i_n_blocking*i_ldc*i_datatype_size) - (i_m*i_datatype_size) );
  }
  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_b, (i_n_blocking*i_ldb*i_datatype_size) );
  libxsmm_instruction_alu_imm( io_generated_code, "subq", i_gp_reg_a, (i_m*i_datatype_size) );
  libxsmm_instruction_alu_imm( io_generated_code, "cmpq", i_gp_reg_nloop, i_n );
  sprintf( l_new_code, "1%ib", i_n_blocking );
  libxsmm_instruction_jump_to_label( io_generated_code, "jl", l_new_code );  
#if 0
  codestream << "                         \"addq $" << ((n_blocking)*ldc * 8) - (M * 8) << ", %%r10\\n\\t\"" << std::endl;
  if (tPrefetch.compare("BL2viaC") == 0) {
    codestream << "                         \"addq $" << ((n_blocking)*ldc * 8) - (M * 8) << ", %%r12\\n\\t\"" << std::endl;
  }
  codestream << "                         \"addq $" << n_blocking* ldb * 8 << ", %%r8\\n\\t\"" << std::endl;
  codestream << "                         \"subq $" << M * 8 << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"cmpq $" << N << ", %%r15\\n\\t\"" << std::endl;
  codestream << "                         \"jl 1" << n_blocking << "b\\n\\t\"" << std::endl;
#endif
}


void libxsmm_generator_dense_header_mloop(char**             io_generated_code,
                                          const unsigned int i_gp_reg_mloop,
                                          const unsigned int i_m_blocking ) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  sprintf( l_new_code, "100%i", i_m_blocking );
  libxsmm_instruction_register_jump_label( io_generated_code, l_new_code );
  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_mloop, i_m_blocking );
#if 0
  codestream << "                         \"100" << m_blocking << ":\\n\\t\"" << std::endl;
  codestream << "                         \"addq $" << m_blocking << ", %%r14\\n\\t\"" << std::endl;
#endif
}

void libxsmm_generator_dense_footer_mloop(char**             io_generated_code,
                                          const unsigned int i_gp_reg_a,
                                          const unsigned int i_gp_reg_c,
                                          const unsigned int i_gp_reg_mloop,
                                          const unsigned int i_m_blocking,
                                          const unsigned int i_m,
                                          const unsigned int i_k,
                                          const unsigned int i_lda,
                                          const char*        i_prefetch,
                                          const unsigned int i_gp_reg_pre_b,
                                          const unsigned int i_datatype_size) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_c, i_m_blocking*i_datatype_size );
  if ( strcmp( i_prefetch, "BL2viaC") == 0 ) {
    libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_pre_b, i_m_blocking*i_datatype_size );
  }
  libxsmm_instruction_alu_imm( io_generated_code, "subq", i_gp_reg_a, (i_k * i_datatype_size * i_lda) - (i_m_blocking * i_datatype_size) );
  libxsmm_instruction_alu_imm( io_generated_code, "cmpq", i_gp_reg_mloop, i_m );
  sprintf( l_new_code, "100%ib", i_m_blocking );
  libxsmm_instruction_jump_to_label( io_generated_code, "jl", l_new_code );  
#if 0
  codestream << "                         \"addq $" << m_blocking * 8 << ", %%r10\\n\\t\"" << std::endl;
  if (tPrefetch.compare("BL2viaC") == 0) {
    codestream << "                         \"addq $" << m_blocking * 8 << ", %%r12\\n\\t\"" << std::endl;
  }
  codestream << "                         \"subq $" << (K * 8 * lda) - (m_blocking * 8) << ", %%r9\\n\\t\"" << std::endl;
  codestream << "                         \"cmpq $" << M_done << ", %%r14\\n\\t\"" << std::endl;
  codestream << "                         \"jl 100" << m_blocking << "b\\n\\t\"" << std::endl;
#endif
}

