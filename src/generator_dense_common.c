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

void libxsmm_generator_dense_sse_avx_open_kernel( char**                        io_generated_code,
                                                  const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                                  const char*                   i_prefetch) { 
  if ( io_generated_code != NULL ) {
    char l_new_code[512];
    char l_gp_reg_name[4];
    
    /* loading b pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b, l_gp_reg_name );
    sprintf( l_new_code, "  __asm__ __volatile__(\"movq %%0, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_string( io_generated_code, l_new_code );

    /* loading a pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a, l_gp_reg_name );
    sprintf( l_new_code, "                       \"movq %%1, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_string( io_generated_code, l_new_code );

    /* loading c pointer in assembley */
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_c, l_gp_reg_name );
    sprintf( l_new_code, "                       \"movq %%2, %%%%%s\\n\\t\"\n", l_gp_reg_name );
    libxsmm_append_string( io_generated_code, l_new_code );

    /* loading b prefetch pointer in assembly */
    if ( ( strcmp(i_prefetch, "BL2viaC") == 0 ) || 
         ( strcmp(i_prefetch, "BL1viaC") == 0 )    ) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_name );
      sprintf( l_new_code, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_string( io_generated_code, l_new_code );
    /* loading a prefetch pointer in assembly */
    } else if ( ( strcmp(i_prefetch, "AL1viaC") == 0 ) ||
                ( strcmp(i_prefetch, "AL2") == 0 )        ) {
      libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_name );
      sprintf( l_new_code, "                       \"movq %%3, %%%%%s\\n\\t\"\n", l_gp_reg_name );
      libxsmm_append_string( io_generated_code, l_new_code );
    } else {}
  } else {
    /* TODO-Greg: how do we interface here? */
    /* this is start of the xGEMM kernel, the registers are in the variables */
  }

  /* reset loop counters */
  libxsmm_instruction_alu_imm( io_generated_code, "movq", i_gp_reg_mapping->gp_reg_mloop, 0);
  libxsmm_instruction_alu_imm( io_generated_code, "movq", i_gp_reg_mapping->gp_reg_nloop, 0);
  libxsmm_instruction_alu_imm( io_generated_code, "movq", i_gp_reg_mapping->gp_reg_kloop, 0);
}

void libxsmm_generator_dense_sse_avx_close_kernel( char**                        io_generated_code,
                                                   const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                                   const char*                   i_prefetch) {
  if ( io_generated_code != NULL) {
    char l_new_code[512];
    char l_gp_reg_a[4];
    char l_gp_reg_b[4];
    char l_gp_reg_c[4];
    char l_gp_reg_pre_a[4];
    char l_gp_reg_pre_b[4];
    char l_gp_reg_mloop[4];
    char l_gp_reg_nloop[4];
    char l_gp_reg_kloop[4];

    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a, l_gp_reg_a );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b, l_gp_reg_b );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_c, l_gp_reg_c );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_a_prefetch, l_gp_reg_pre_a );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_b_prefetch, l_gp_reg_pre_b );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_mloop, l_gp_reg_mloop );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_nloop, l_gp_reg_nloop );
    libxsmm_get_x86_gp_reg_name( i_gp_reg_mapping->gp_reg_kloop, l_gp_reg_kloop );

    if ( ( strcmp(i_prefetch, "BL2viaC") == 0 ) || 
         ( strcmp(i_prefetch, "BL1viaC") == 0 )    ) {
      sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(B_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_b, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      libxsmm_append_string( io_generated_code, l_new_code );
    } else if ( ( strcmp(i_prefetch, "AL1viaC") == 0 ) ||
                ( strcmp(i_prefetch, "AL2") == 0 )        ) {
      sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C), \"m\"(A_prefetch) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_pre_a, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      libxsmm_append_string( io_generated_code, l_new_code );
    } else {
      sprintf( l_new_code, "                       : : \"m\"(B), \"m\"(A), \"m\"(C) : \"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"%s\",\"xmm0\",\"xmm1\",\"xmm2\",\"xmm3\",\"xmm4\",\"xmm5\",\"xmm6\",\"xmm7\",\"xmm8\",\"xmm9\",\"xmm10\",\"xmm11\",\"xmm12\",\"xmm13\",\"xmm14\",\"xmm15\");\n", l_gp_reg_a, l_gp_reg_b, l_gp_reg_c, l_gp_reg_mloop, l_gp_reg_nloop, l_gp_reg_kloop);
      libxsmm_append_string( io_generated_code, l_new_code );
    }
  } else {
    /* TODO-Greg: how do we interface here? */
  }
}

void libxsmm_generator_dense_header_kloop(char**                        io_generated_code,
                                          const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                          const unsigned int            i_m_blocking,
                                          const unsigned int            i_k_blocking) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  libxsmm_instruction_alu_imm( io_generated_code, "movq", i_gp_reg_mapping->gp_reg_kloop, 0);
  sprintf( l_new_code, "2%i", i_m_blocking );
  libxsmm_instruction_register_jump_label( io_generated_code, l_new_code );
  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_mapping->gp_reg_kloop, i_k_blocking);
}

void libxsmm_generator_dense_footer_kloop(char**                        io_generated_code,
                                          const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                          const unsigned int            i_m_blocking,
                                          const unsigned int            i_k,
                                          const unsigned int            i_datatype_size,
                                          const unsigned int            i_kloop_complete ) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  libxsmm_instruction_alu_imm( io_generated_code, "cmpq", i_gp_reg_mapping->gp_reg_kloop, i_k );
  sprintf( l_new_code, "2%ib", i_m_blocking );
  libxsmm_instruction_jump_to_label( io_generated_code, "jl", l_new_code );
  if ( i_kloop_complete != 0 ) {
    libxsmm_instruction_alu_imm( io_generated_code, "subq", i_gp_reg_mapping->gp_reg_b, i_k*i_datatype_size );
  }
}

void libxsmm_generator_dense_header_nloop(char**                        io_generated_code,
                                          const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                          const unsigned int            i_n_blocking) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  sprintf( l_new_code, "1%i", i_n_blocking );
  libxsmm_instruction_register_jump_label( io_generated_code, l_new_code );
  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_mapping->gp_reg_nloop, i_n_blocking );
  libxsmm_instruction_alu_imm( io_generated_code, "movq", i_gp_reg_mapping->gp_reg_mloop, 0 );
}


void libxsmm_generator_dense_footer_nloop(char**                        io_generated_code,
                                          const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                          const unsigned int            i_n_blocking,
                                          const unsigned int            i_m,
                                          const unsigned int            i_n,
                                          const unsigned int            i_ldb,
                                          const unsigned int            i_ldc,
                                          const char*                   i_prefetch,
                                          const unsigned int            i_datatype_size) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_mapping->gp_reg_c, (i_n_blocking*i_ldc*i_datatype_size) - (i_m*i_datatype_size) );
  if ( strcmp( i_prefetch, "BL2viaC") == 0 ) {
    libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_mapping->gp_reg_b_prefetch, (i_n_blocking*i_ldc*i_datatype_size) - (i_m*i_datatype_size) );
  }
  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_mapping->gp_reg_b, (i_n_blocking*i_ldb*i_datatype_size) );
  libxsmm_instruction_alu_imm( io_generated_code, "subq", i_gp_reg_mapping->gp_reg_a, (i_m*i_datatype_size) );
  libxsmm_instruction_alu_imm( io_generated_code, "cmpq", i_gp_reg_mapping->gp_reg_nloop, i_n );
  sprintf( l_new_code, "1%ib", i_n_blocking );
  libxsmm_instruction_jump_to_label( io_generated_code, "jl", l_new_code );  
}


void libxsmm_generator_dense_header_mloop(char**                        io_generated_code,
                                          const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                          const unsigned int            i_m_blocking ) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  sprintf( l_new_code, "100%i", i_m_blocking );
  libxsmm_instruction_register_jump_label( io_generated_code, l_new_code );
  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_mapping->gp_reg_mloop, i_m_blocking );
}

void libxsmm_generator_dense_footer_mloop(char**                        io_generated_code,
                                          const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                          const unsigned int            i_m_blocking,
                                          const unsigned int            i_m,
                                          const unsigned int            i_k,
                                          const unsigned int            i_lda,
                                          const char*                   i_prefetch,
                                          const unsigned int            i_datatype_size) {
  char l_new_code[32];
  l_new_code[0] = '\0';

  libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_mapping->gp_reg_c, i_m_blocking*i_datatype_size );
  if ( strcmp( i_prefetch, "BL2viaC") == 0 ) {
    libxsmm_instruction_alu_imm( io_generated_code, "addq", i_gp_reg_mapping->gp_reg_b_prefetch, i_m_blocking*i_datatype_size );
  }
  libxsmm_instruction_alu_imm( io_generated_code, "subq", i_gp_reg_mapping->gp_reg_a, (i_k * i_datatype_size * i_lda) - (i_m_blocking * i_datatype_size) );
  libxsmm_instruction_alu_imm( io_generated_code, "cmpq", i_gp_reg_mapping->gp_reg_mloop, i_m );
  sprintf( l_new_code, "100%ib", i_m_blocking );
  libxsmm_instruction_jump_to_label( io_generated_code, "jl", l_new_code );  
}

