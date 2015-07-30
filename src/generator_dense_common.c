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

void libxsmm_generator_dense_init_micro_kernel_config_fullvector( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                  const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                                                  const char*                     i_arch,
                                                                  const unsigned int              i_use_masking_a_c ) {
  if( strcmp( i_arch, "snb" ) == 0 ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_X86_AVX;
    fprintf(stderr, "LIBXSMM ERROR, ibxsmm_generator_dense_init_micro_kernel_config_fullvector, unsupported architecture!!!\n");
    exit(-1);
  } else if ( strcmp( i_arch, "hsw" ) == 0 ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_X86_AVX2;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'y';
    if ( i_xgemm_desc->single_precision == 0 ) {  
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size = 8;
      if ( i_xgemm_desc->aligned_a != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( i_xgemm_desc->aligned_c != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    } else {
      io_micro_kernel_config->vector_length = 8;
      io_micro_kernel_config->datatype_size = 4;
      if ( i_xgemm_desc->aligned_a != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( i_xgemm_desc->aligned_c != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    }
  } else {
    fprintf(stderr, "LIBXSMM ERROR, ibxsmm_generator_dense_init_micro_kernel_config_fullvector, unsupported architecture!!!\n");
    exit(-1);
  }

  io_micro_kernel_config->prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCH1;
  io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL; 
  io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ; 
}

void libxsmm_generator_dense_init_micro_kernel_config_halfvector( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                                  const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                                                  const char*                     i_arch,
                                                                  const unsigned int              i_use_masking_a_c ) {
  if( strcmp( i_arch, "snb" ) == 0 ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_X86_AVX;
    fprintf(stderr, "LIBXSMM ERROR, ibxsmm_generator_dense_init_micro_kernel_config_fullvector, unsupported architecture!!!\n");
    exit(-1);
  } else if ( strcmp( i_arch, "hsw" ) == 0 ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_X86_AVX2;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'y';
    if ( i_xgemm_desc->single_precision == 0 ) {  
      io_micro_kernel_config->vector_length = 2;
      io_micro_kernel_config->datatype_size = 8;
      if ( i_xgemm_desc->aligned_a != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( i_xgemm_desc->aligned_c != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPD;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPD;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    } else {
      io_micro_kernel_config->vector_length = 4;
      io_micro_kernel_config->datatype_size = 4;
      if ( i_xgemm_desc->aligned_a != 0 ) {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      } else {
        io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      }
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VBROADCASTSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      if ( i_xgemm_desc->aligned_c != 0 ) {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVUPS;
      } else {
        io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVAPS;
      }
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPS;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231PS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    }
  } else {
    fprintf(stderr, "LIBXSMM ERROR, ibxsmm_generator_dense_init_micro_kernel_config_fullvector, unsupported architecture!!!\n");
    exit(-1);
  }

  io_micro_kernel_config->prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCH1;
  io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL; 
  io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ; 
}

void libxsmm_generator_dense_init_micro_kernel_config_scalar( libxsmm_micro_kernel_config*    io_micro_kernel_config,
                                                              const libxsmm_xgemm_descriptor* i_xgemm_desc,
                                                              const char*                     i_arch,
                                                              const unsigned int              i_use_masking_a_c ) {
  if( strcmp( i_arch, "snb" ) == 0 ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_X86_AVX;
    fprintf(stderr, "LIBXSMM ERROR, ibxsmm_generator_dense_init_micro_kernel_config_fullvector, unsupported architecture!!!\n");
    exit(-1);
  } else if ( strcmp( i_arch, "hsw" ) == 0 ) {
    io_micro_kernel_config->instruction_set = LIBXSMM_X86_AVX2;
    io_micro_kernel_config->use_masking_a_c = i_use_masking_a_c;
    io_micro_kernel_config->vector_name = 'y';
    if ( i_xgemm_desc->single_precision == 0 ) {  
      io_micro_kernel_config->vector_length = 1;
      io_micro_kernel_config->datatype_size = 8;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSD;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSD;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSD;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231SD;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    } else {
      io_micro_kernel_config->vector_length = 1;
      io_micro_kernel_config->datatype_size = 4;
      io_micro_kernel_config->a_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSS;
      io_micro_kernel_config->b_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSS;
      io_micro_kernel_config->b_shuff_instruction = LIBXSMM_X86_INSTR_UNDEF;
      io_micro_kernel_config->c_vmove_instruction = LIBXSMM_X86_INSTR_VMOVSS;
      io_micro_kernel_config->vxor_instruction = LIBXSMM_X86_INSTR_VXORPD;
      io_micro_kernel_config->vmul_instruction = LIBXSMM_X86_INSTR_VFMADD231SS;
      io_micro_kernel_config->vadd_instruction = LIBXSMM_X86_INSTR_UNDEF;
    }
  } else {
    fprintf(stderr, "LIBXSMM ERROR, ibxsmm_generator_dense_init_micro_kernel_config_fullvector, unsupported architecture!!!\n");
    exit(-1);
  }

  io_micro_kernel_config->prefetch_instruction = LIBXSMM_X86_INSTR_PREFETCH1;
  io_micro_kernel_config->alu_add_instruction = LIBXSMM_X86_INSTR_ADDQ;
  io_micro_kernel_config->alu_sub_instruction = LIBXSMM_X86_INSTR_SUBQ;
  io_micro_kernel_config->alu_cmp_instruction = LIBXSMM_X86_INSTR_CMPQ;
  io_micro_kernel_config->alu_jmp_instruction = LIBXSMM_X86_INSTR_JL; 
  io_micro_kernel_config->alu_mov_instruction = LIBXSMM_X86_INSTR_MOVQ; 
}

void libxsmm_generator_dense_signature( libxsmm_generated_code*         io_generated_code,
                                        const char*                     i_routine_name,
                                        const libxsmm_xgemm_descriptor* i_xgemm_desc ) {
  char l_new_code_line[512];

  if ( io_generated_code->code_type != 0 )
    return;
  
  /* selecting the correct signature */
  if (i_xgemm_desc->single_precision == 1) {
    if ( strcmp(i_xgemm_desc->prefetch, "nopf") == 0) {
      sprintf(l_new_code_line, "void %s(const float* A, const float* B, float* C) {\n", i_routine_name);
    } else {
      sprintf(l_new_code_line, "void %s(const float* A, const float* B, float* C, const float* A_prefetch = NULL, const float* B_prefetch = NULL, const float* C_prefetch = NULL) {\n", i_routine_name);
    }
  } else {
    if ( strcmp(i_xgemm_desc->prefetch, "nopf") == 0) {
      sprintf(l_new_code_line, "void %s(const double* A, const double* B, double* C) {\n", i_routine_name);
    } else {
      sprintf(l_new_code_line, "void %s(const double* A, const double* B, double* C, const double* A_prefetch = NULL, const double* B_prefetch = NULL, const double* C_prefetch = NULL) {\n", i_routine_name);
    }
  }

  libxsmm_append_code_as_string( io_generated_code, l_new_code_line );
}

void libxsmm_generator_dense_add_isa_check_header( libxsmm_generated_code* io_generated_code, 
                                                   const char*             i_arch ) {
  if ( io_generated_code->code_type != 0 )
    return;

  if ( (strcmp( i_arch, "wsm" ) == 0) ) {
    libxsmm_append_code_as_string( io_generated_code, "#ifdef __SSE3__\n");
    libxsmm_append_code_as_string( io_generated_code, "#ifdef __AVX__\n");
    libxsmm_append_code_as_string( io_generated_code, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling SSE3 code on AVX or newer architecture: \" __FILE__)\n");
    libxsmm_append_code_as_string( io_generated_code, "#endif\n");
  } else if ( (strcmp( i_arch, "snb" ) == 0) ) {
    libxsmm_append_code_as_string( io_generated_code, "#ifdef __AVX__\n");
    libxsmm_append_code_as_string( io_generated_code, "#ifdef __AVX2__\n");
    libxsmm_append_code_as_string( io_generated_code, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling AVX code on AVX2 or newer architecture: \" __FILE__)\n");
    libxsmm_append_code_as_string( io_generated_code, "#endif\n");
  } else if ( (strcmp( i_arch, "hsw" ) == 0) ) {
    libxsmm_append_code_as_string( io_generated_code, "#ifdef __AVX2__\n");
    libxsmm_append_code_as_string( io_generated_code, "#ifdef __AVX512F__\n");
    libxsmm_append_code_as_string( io_generated_code, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling AVX2 code on AVX512 or newer architecture: \" __FILE__)\n");
    libxsmm_append_code_as_string( io_generated_code, "#endif\n");
  } else if ( (strcmp( i_arch, "knl" ) == 0) ||
              (strcmp( i_arch, "skx" ) == 0) ) {
    libxsmm_append_code_as_string( io_generated_code, "#ifdef __AVX512F__\n");
  } else if ( (strcmp( i_arch, "noarch" ) == 0) ) {
    libxsmm_append_code_as_string( io_generated_code, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling arch-independent gemm kernel in: \" __FILE__)\n");
  } else {
    fprintf(stderr, "unsupported architecture in libxsmm_generator_dense_add_isa_check_header\n");
    exit(-1);   
  }
}

void libxsmm_generator_dense_add_isa_check_footer( libxsmm_generated_code* io_generated_code, 
                                                   const char*             i_arch ) {
  if ( io_generated_code->code_type != 0 )
    return;

  if ( (strcmp( i_arch, "wsm" ) == 0) || 
       (strcmp( i_arch, "snb" ) == 0) || 
       (strcmp( i_arch, "hsw" ) == 0) || 
       (strcmp( i_arch, "knc" ) == 0) || 
       (strcmp( i_arch, "knl" ) == 0) || 
       (strcmp( i_arch, "skx" ) == 0)    ) {
    libxsmm_append_code_as_string( io_generated_code, "#else\n");
    libxsmm_append_code_as_string( io_generated_code, "#pragma message (\"LIBXSMM KERNEL COMPILATION ERROR in: \" __FILE__)\n");
    libxsmm_append_code_as_string( io_generated_code, "#error No kernel was compiled, lacking support for current architecture?\n");
    libxsmm_append_code_as_string( io_generated_code, "#endif\n\n");
  } else if ( (strcmp( i_arch, "noarch" ) == 0) ) {
  } else {
    fprintf(stderr, "unsupported architecture in libxsmm_generator_dense_add_isa_check_footer\n");
    exit(-1);   
  }
}

void libxsmm_generator_dense_add_flop_counter( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_xgemm_descriptor* i_xgemm_desc ) {
  if ( io_generated_code->code_type != 0 )
    return;

  char l_new_code[512];
  
  libxsmm_append_code_as_string( io_generated_code, "#ifndef NDEBUG\n");
  libxsmm_append_code_as_string( io_generated_code, "#ifdef _OPENMP\n");
  libxsmm_append_code_as_string( io_generated_code, "#pragma omp atomic\n");
  libxsmm_append_code_as_string( io_generated_code, "#endif\n");
  sprintf( l_new_code, "libxsmm_num_total_flops += %i;\n", 2 * i_xgemm_desc->m * i_xgemm_desc->n * i_xgemm_desc->k);
  libxsmm_append_code_as_string( io_generated_code, l_new_code);
  libxsmm_append_code_as_string( io_generated_code, "#endif\n\n");
}

void libxsmm_generator_dense_header_kloop( libxsmm_generated_code*             io_generated_code,
                                           const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                           const unsigned int                  i_m_blocking,
                                           const unsigned int                  i_k_blocking ) {
  char l_new_code[32];

  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_kloop, 0);
  sprintf( l_new_code, "30%i", i_m_blocking );
  libxsmm_instruction_register_jump_label( io_generated_code, l_new_code );
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_kloop, i_k_blocking);
}

void libxsmm_generator_dense_footer_kloop( libxsmm_generated_code*             io_generated_code,
                                           const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                           const libxsmm_xgemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                  i_m_blocking,
                                           const unsigned int                  i_max_blocked_k,
                                           const unsigned int                  i_kloop_complete ) {
  char l_new_code[32];

  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_kloop, i_max_blocked_k );
  sprintf( l_new_code, "30%ib", i_m_blocking );
  libxsmm_instruction_jump_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, l_new_code );
  if ( i_kloop_complete != 0 ) {
    libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, 
                                 i_gp_reg_mapping->gp_reg_b, (i_xgemm_desc->k)*(i_micro_kernel_config->datatype_size) );
  }
}

void libxsmm_generator_dense_header_nloop( libxsmm_generated_code*             io_generated_code,
                                           const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                           const unsigned int                  i_n_blocking) {
  char l_new_code[32];

  sprintf( l_new_code, "1%i", i_n_blocking );
  libxsmm_instruction_register_jump_label( io_generated_code, l_new_code );
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_nloop, i_n_blocking );
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_mov_instruction, i_gp_reg_mapping->gp_reg_mloop, 0 );
}


void libxsmm_generator_dense_footer_nloop( libxsmm_generated_code*             io_generated_code,
                                           const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                           const libxsmm_xgemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                  i_n_blocking ) {
  char l_new_code[32];

  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_c, 
                               (i_n_blocking*(i_xgemm_desc->ldc)*(i_micro_kernel_config->datatype_size)) - ((i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size)) );
  if ( strcmp( i_xgemm_desc->prefetch, "BL2viaC") == 0 ) {
    libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_b_prefetch, 
                                 (i_n_blocking*(i_xgemm_desc->ldc)*(i_micro_kernel_config->datatype_size)) - ((i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size)) );
  }
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, 
                               i_gp_reg_mapping->gp_reg_b, (i_n_blocking*(i_xgemm_desc->ldb)*(i_micro_kernel_config->datatype_size)) );
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, 
                               i_gp_reg_mapping->gp_reg_a, ((i_xgemm_desc->m)*(i_micro_kernel_config->datatype_size)) );
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_nloop, i_xgemm_desc->n );
  sprintf( l_new_code, "1%ib", i_n_blocking );
  libxsmm_instruction_jump_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, l_new_code );  
}


void libxsmm_generator_dense_header_mloop( libxsmm_generated_code*             io_generated_code,
                                           const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                           const unsigned int                  i_m_blocking ) {
  char l_new_code[32];

  sprintf( l_new_code, "20%i", i_m_blocking );
  libxsmm_instruction_register_jump_label( io_generated_code, l_new_code );
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, i_gp_reg_mapping->gp_reg_mloop, i_m_blocking );
}

void libxsmm_generator_dense_footer_mloop( libxsmm_generated_code*             io_generated_code,
                                           const libxsmm_gp_reg_mapping*       i_gp_reg_mapping,
                                           const libxsmm_micro_kernel_config*  i_micro_kernel_config,
                                           const libxsmm_xgemm_descriptor*     i_xgemm_desc,
                                           const unsigned int                  i_m_blocking,
                                           const unsigned int                  i_m_done ) {
  char l_new_code[32];

  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, 
                               i_gp_reg_mapping->gp_reg_c, i_m_blocking*(i_micro_kernel_config->datatype_size) );
  if ( strcmp( i_xgemm_desc->prefetch, "BL2viaC") == 0 ) {
    libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_add_instruction, 
                                 i_gp_reg_mapping->gp_reg_b_prefetch, i_m_blocking*(i_micro_kernel_config->datatype_size) );
  }
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_sub_instruction, i_gp_reg_mapping->gp_reg_a, 
                               ((i_xgemm_desc->k) * (i_micro_kernel_config->datatype_size) * (i_xgemm_desc->lda) ) - (i_m_blocking * (i_micro_kernel_config->datatype_size)) );
  libxsmm_instruction_alu_imm( io_generated_code, i_micro_kernel_config->alu_cmp_instruction, i_gp_reg_mapping->gp_reg_mloop, i_m_done );
  sprintf( l_new_code, "20%ib", i_m_blocking );
  libxsmm_instruction_jump_to_label( io_generated_code, i_micro_kernel_config->alu_jmp_instruction, l_new_code );  
}

