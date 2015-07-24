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

#ifndef GENERATOR_DENSE_INSTRUCTIONS_H
#define GENERATOR_DENSE_INSTRUCTIONS_H

#include "generator_common.h"

/**
 * Opens the inline assembly section / jit stream
 *
 * @param io_generated_code pointer to the pointer of the generated code buffer
 * @param i_gp_reg_mapping gp register mapping for initialization
 * @param i_prefetch prefetch mode which may result in addtional gp reg inits
 */
void libxsmm_generator_dense_sse_avx_open_instrucion_stream( char**                        io_generated_code,
                                                             const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                                             const char*                   i_prefetch);

/**
 * Closes the inline assembly section / jit stream
 *
 * @param io_generated_code pointer to the pointer of the generated code buffer
 * @param i_gp_reg_mapping gp register mapping for clobbering
 * @param i_prefetch prefetch mode which may result in addtional gp reg clobbers
 */
void libxsmm_generator_dense_sse_avx_close_instruction_stream( char**                        io_generated_code,
                                                               const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                                               const char*                   i_prefetch);

/**
 * Generates vmovapd/vmovupd/vmovaps/vmovups/vmovsd/vmovss/vbroadcastsd/vbroastcastss/vmovddup instructions with displacements, explicit SIB addressing is not
 * supported by this function
 *
 * @param io_generated_code pointer to the pointer of the generated code buffer
 * @param i_vmove_instr actual vmov variant
 * @param i_gp_reg_number the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_displacement the offset to the base address 
 * @param i_vector_name the vector register name prefix (xmm,ymm or zmm)
 * @param i_vec_reg_number_0 the vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_is_store 0: load semantik, other: store semantik  
 */
void libxsmm_instruction_vec_move( char**             io_generated_code, 
                                   const char*        i_vmove_instr, 
                                   const unsigned int i_gp_reg_number,
                                   const int          i_displacement,
                                   const char*        i_vector_name,
                                   const unsigned int i_vec_reg_number_0,
                                   const unsigned int i_is_store );

/* @TODO this doesn't work right now for SSE */
/**
 * Generates (v)XYZpd/(v)XYZps/(v)XYZsd/(v)XYZss instructions with 2 or 3 vector registers, memory operands are not supported as first operand
 *
 * @param io_generated_code pointer to the pointer of the generated code buffer
 * @param i_vec_instr actual operation variant
 * @param i_vector_name the vector register name prefix (xmm,ymm or zmm)
 * @param i_vec_reg_number_0 the first vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_vec_reg_number_1 the first vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_vec_reg_number_2 the first vector register number (xmm/ymm: 0-15, zmm: 0-31)
 */
void libxsmm_instruction_vec_compute_reg( char**             io_generated_code, 
                                          const char*        i_vec_instr,
                                          const char*        i_vector_name,                                
                                          const unsigned int i_vec_reg_number_0,
                                          const unsigned int i_vec_reg_number_1,
                                          const unsigned int i_vec_reg_number_2 );

/* @TODO check if we can merge this alu_imm */
/**
 * Generates prefetch instructions with displacements, SIB addressing is not
 * supported by this function
 *
 * @param io_generated_code pointer to the pointer of the generated code buffer
 * @param i_prefetch_instr actual prefetch variant
 * @param i_gp_reg_number the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_displacement the offset to the base address 
 */
void libxsmm_instruction_prefetch( char**             io_generated_code,
                                   const char*        i_prefetch_instr, 
                                   const unsigned int i_gp_reg_number,
                                   const int          i_displacement);

/**
 * Generates regular all instructions with immediates
 *
 * @param io_generated_code pointer to the pointer of the generated code buffer
 * @param i_alu_instr actual alu gpr instruction
 * @param i_gp_reg_number the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_immediate the immediate operand 
 */
void libxsmm_instruction_alu_imm( char**             io_generated_code,
                                  const char*        i_alu_instr,
                                  const unsigned int i_gp_reg_number,
                                  const unsigned int i_immediate);

/**
 * Generates regular all instructions with immediates
 *
 * @param io_generated_code pointer to the pointer of the generated code buffer
 * @param i_jmp_label jump label that is insert into the code
*/
void libxsmm_instruction_register_jump_label( char**      io_generated_code,
                                              const char* i_jmp_label );

/**
 * Generates regular all instructions with immediates
 *
 * @param io_generated_code pointer to the pointer of the generated code buffer
 * @param i_jmp_instr the particular jump instruction used
 * @param i_jmp_label jump label that is the target of the jump, THIS IS NOT CHECKED, YOU TO ENSURE THAT IT EXISTS IN ORDER TO AVOID ASSEMBLER ERRORS
*/
void libxsmm_instruction_jump_to_label( char**      io_generated_code,
                                        const char* i_jmp_instr,
                                        const char* i_jmp_label );

#endif /* GENERATOR_DENSE_INSTRUCTIONS_H */

