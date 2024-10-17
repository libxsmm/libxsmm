/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Greg Henry (Intel Corp.)
******************************************************************************/
#ifndef GENERATOR_X86_INSTRUCTIONS_H
#define GENERATOR_X86_INSTRUCTIONS_H

#include "generator_common.h"

typedef enum libxsmm_x86_simd_name {
  LIBXSMM_X86_SIMD_NAME_XMM = 0x0,
  LIBXSMM_X86_SIMD_NAME_YMM = 0x1,
  LIBXSMM_X86_SIMD_NAME_ZMM = 0x2
} libxsmm_x86_simd_name;

LIBXSMM_API_INTERN
unsigned int libxsmm_x86_instruction_vec_is_hybrid( const unsigned int i_instr );

LIBXSMM_API_INTERN
unsigned int libxsmm_x86_instruction_vec_is_regmemonly( const unsigned int i_instr );

LIBXSMM_API_INTERN
unsigned int libxsmm_x86_instruction_vec_is_regonly( const unsigned int i_instr );

/**
 * Opens the inline assembly section / jit stream
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gp_reg_mapping gp register mapping for initialization
 * @param i_prefetch prefetch mode which may result in additional gp reg inits
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream_gemm( libxsmm_generated_code*       io_generated_code,
                                               const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                               const unsigned int            skip_callee_save,
                                               unsigned int                  i_prefetch );

/**
 * Closes the inline assembly section / jit stream
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gp_reg_mapping gp register mapping for clobbering
 * @param i_prefetch prefetch mode which may result in additional gp reg clobbers
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream_gemm( libxsmm_generated_code*       io_generated_code,
                                                const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                                const unsigned int            skip_callee_save,
                                                unsigned int                  i_prefetch );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream_alt( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_gp_struct_params,
                                              const unsigned int      skip_callee_save );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream_alt( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      skip_callee_save );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_lea_data( libxsmm_generated_code*     io_generated_code,
                                       unsigned int                i_reg,
                                       unsigned int                i_off,
                                       libxsmm_const_data_tracker* io_const_data );

LIBXSMM_API_INTERN
unsigned int libxsmm_x86_instruction_add_data( libxsmm_generated_code*     io_generated_code,
                                               const unsigned char*        i_data,
                                               unsigned int                i_ndata_bytes,
                                               unsigned int                i_alignment,
                                               unsigned int                i_append_only,
                                               libxsmm_const_data_tracker* io_const_data );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_data( libxsmm_generated_code*     io_generated_code,
                                         libxsmm_const_data_tracker* io_const_data );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_rex_compute_1reg_mem( libxsmm_generated_code*     io_generated_code,
                                                   const unsigned int          i_instr,
                                                   const unsigned int          i_gp_reg_base,
                                                   const unsigned int          i_gp_reg_idx,
                                                   const unsigned int          i_scale,
                                                   const int                   i_displacement,
                                                   const unsigned int          i_reg_number_reg );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_rex_compute_2reg( libxsmm_generated_code*     io_generated_code,
                                               const unsigned int          i_instr,
                                               const unsigned int          i_reg_number_rm,
                                               const unsigned int          i_reg_number_reg );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vex_compute_2reg_mem( libxsmm_generated_code*     io_generated_code,
                                                   const unsigned int          i_vec_instr,
                                                   const unsigned int          i_gp_reg_base,
                                                   const unsigned int          i_gp_reg_idx,
                                                   const unsigned int          i_scale,
                                                   const int                   i_displacement,
                                                   const libxsmm_x86_simd_name i_vector_name,
                                                   const unsigned int          i_vec_reg_number_src,
                                                   const unsigned int          i_vec_reg_number_dst );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vex_compute_3reg( libxsmm_generated_code*     io_generated_code,
                                               const unsigned int          i_vec_instr,
                                               const libxsmm_x86_simd_name i_vector_name,
                                               const unsigned int          i_vec_reg_number_0,
                                               const unsigned int          i_vec_reg_number_1,
                                               const unsigned int          i_vec_reg_number_2 );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_evex_compute_2reg_mem( libxsmm_generated_code*     io_generated_code,
                                                    const unsigned int          i_vec_instr,
                                                    const unsigned int          i_use_broadcast,
                                                    const unsigned int          i_gp_reg_base,
                                                    const unsigned int          i_reg_idx,
                                                    const unsigned int          i_scale,
                                                    const int                   i_displacement,
                                                    const libxsmm_x86_simd_name i_vector_name,
                                                    const unsigned int          i_vec_reg_number_src,
                                                    const unsigned int          i_vec_reg_number_dst,
                                                    const unsigned int          i_mask_reg_number,
                                                    const unsigned int          i_use_zero_masking );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_evex_compute_3reg( libxsmm_generated_code*     io_generated_code,
                                                const unsigned int          i_vec_instr,
                                                const libxsmm_x86_simd_name i_vector_name,
                                                const unsigned int          i_vec_reg_number_0,
                                                const unsigned int          i_vec_reg_number_1,
                                                const unsigned int          i_vec_reg_number_2,
                                                const unsigned int          i_mask_reg_number,
                                                const unsigned int          i_use_zero_masking,
                                                const unsigned char         i_sae_cntl );

/**
 * Generates vmaskmovps/vmaskmovpd/vgathers with displacements for loads and stores.
 * Only works with i_vector_name='Y'
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_mask_move( libxsmm_generated_code* io_generated_code,
                                            const unsigned int      i_vmove_instr,
                                            const unsigned int      i_gp_reg_base,
                                            const unsigned int      i_reg_idx,
                                            const unsigned int      i_scale,
                                            const int               i_displacement,
                                            const char              i_vector_name,
                                            const unsigned int      i_vec_reg_number_0,
                                            const unsigned int      i_vec_reg_mask_0,
                                            const unsigned int      i_is_store );

/**
 * Generates vmovapd/vmovupd/vmovaps/vmovups/vmovsd/vmovss/vbroadcastsd/vbroastcastss/vmovddup instructions with displacements, explicit SIB addressing is not
 * supported by this function
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_instruction_set requested instruction set to encode
 * @param i_vmove_instr actual vmov variant
 * @param i_gp_reg_base the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_displacement the offset to the base address
 * @param i_vector_name the vector register name prefix (x, y or z)
 * @param i_vec_reg_number_0 the vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_mask_reg_number the mask register to be used
 * @param i_use_zero_masking 0: merge masking ; !=0: zero masking
 * @param i_is_store 0: load semantic, other: store semantic
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_move( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_instruction_set,
                                       const unsigned int      i_vmove_instr,
                                       const unsigned int      i_gp_reg_base,
                                       const unsigned int      i_reg_idx,
                                       const unsigned int      i_scale,
                                       const int               i_displacement,
                                       const char              i_vector_name,
                                       const unsigned int      i_vec_reg_number_0,
                                       const unsigned int      i_mask_reg_number,
                                       const unsigned int      i_use_zero_masking,
                                       const unsigned int      i_is_store );

/**
 * Generates (v)XYZpd/(v)XYZps/(v)XYZsd/(v)XYZss instructions with 3 vector registers and masking
 * it provides a commin interface for REX/VEX/EVEX vector compute instructions
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_instr actual operation variant
 * @param i_vector_name the vector register name prefix (z)
 * @param i_reg_number_src0 the first vector register number (zmm: 0-31)
 * @param i_reg_number_src1 the second vector register number (zmm: 0-31), maybe LIBXSMM_VEC_REG_UNDEF if 2 operand instruction
 * @param i_reg_number_dst the second vector register number (zmm: 0-31), or mask (1-7)
 * @param i_mask_reg_number the mask register to read/write
 * @param i_mask_cntl 0: merge masking, !=0 zero masking
 * @param i_sae_cntl > 0:  bit 0/1: use SAE, bit 1/2: RC, >0 automatically implies 512bit width.
 * @param i_imm8 immediate just as the compare value for a compare instruction
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_3reg_mask_sae_imm8( libxsmm_generated_code* io_generated_code,
                                                             const unsigned int      i_vec_instr,
                                                             const char              i_vector_name,
                                                             const unsigned int      i_reg_number_src0,
                                                             const unsigned int      i_reg_number_src1,
                                                             const unsigned int      i_reg_number_dst,
                                                             const unsigned int      i_mask_reg_number,
                                                             const unsigned int      i_mask_cntl,
                                                             const unsigned char     i_sae_cntl,
                                                             const unsigned int      i_imm8 );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_3reg( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_vec_instr,
                                               const char              i_vector_name,
                                               const unsigned int      i_reg_number_src0,
                                               const unsigned int      i_reg_number_src1,
                                               const unsigned int      i_reg_number_dst );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_3reg_mask( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_instr,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_reg_number_src0,
                                                    const unsigned int      i_reg_number_src1,
                                                    const unsigned int      i_reg_number_dst,
                                                    const unsigned int      i_mask_reg_number,
                                                    const unsigned int      i_mask_cntl );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_3reg_imm8( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_instr,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_reg_number_src0,
                                                    const unsigned int      i_reg_number_src1,
                                                    const unsigned int      i_reg_number_dst,
                                                    const unsigned int      i_imm8 );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_2reg_mask_sae_imm8( libxsmm_generated_code* io_generated_code,
                                                             const unsigned int      i_vec_instr,
                                                             const char              i_vector_name,
                                                             const unsigned int      i_reg_number_src0,
                                                             const unsigned int      i_reg_number_dst,
                                                             const unsigned int      i_mask_reg_number,
                                                             const unsigned int      i_mask_cntl,
                                                             const unsigned char     i_sae_cntl,
                                                             const unsigned int      i_imm8 );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_2reg( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_vec_instr,
                                               const char              i_vector_name,
                                               const unsigned int      i_reg_number_src0,
                                               const unsigned int      i_reg_number_dst );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_2reg_mask( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_instr,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_reg_number_src0,
                                                    const unsigned int      i_reg_number_dst,
                                                    const unsigned int      i_mask_reg_number,
                                                    const unsigned int      i_mask_cntl);

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_2reg_imm8( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_instr,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_reg_number_src0,
                                                    const unsigned int      i_reg_number_dst,
                                                    const unsigned int      i_imm8 );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_1reg_imm8( libxsmm_generated_code* io_generated_code,
                                                    const unsigned int      i_vec_instr,
                                                    const char              i_vector_name,
                                                    const unsigned int      i_reg_number_dst,
                                                    const unsigned int      i_imm8 );

/**
 * Generates (v)XYZpd/(v)XYZps/(v)XYZsd/(v)XYZss instructions with 3 vector registers and masking
 * it provides a commin interface for REX/VEX/EVEX vector compute instructions
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_vec_instr actual operation variant
 * @param i_vector_name the vector register name prefix (z)
 * @param i_gp_reg_base base address register for memory broadcast
 * @param i_gp_reg_idx index register for memory broadcast, can be LIBXSMM_X86_GP_REG_UNDEF -> then regular displacement version is generated
 * @param i_scale scale of index register, ignored if i_gp_reg_idx is LIBXSMM_X86_GP_REG_UNDEF
 * @param i_displacement displacement to SIB address
 * @param i_use_broadcast if != 0 memory operand is interpreted as a scalar and broadcasted in fused fashion, only supported on AVX512
 * @param i_reg_number_src1 the second vector register number (zmm: 0-31), maybe LIBXSMM_VEC_REG_UNDEF if 2 operand instruction
 * @param i_reg_number_dst the second vector register number (zmm: 0-31), or mask (1-7)
 * @param i_mask_reg_number the mask register to read/write
 * @param i_mask_rnd_exp_cntl 0: merge masking, !=0 zero masking
 * @param i_imm8 immediate just as the compare value for a compare instruction
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_2reg_mask_imm8( libxsmm_generated_code* io_generated_code,
                                                             const unsigned int      i_vec_instr,
                                                             const char              i_vector_name,
                                                             const unsigned int      i_gp_reg_base,
                                                             const unsigned int      i_gp_reg_idx,
                                                             const unsigned int      i_scale,
                                                             const int               i_displacement,
                                                             const unsigned int      i_use_broadcast,
                                                             const unsigned int      i_reg_number_src1,
                                                             const unsigned int      i_reg_number_dst,
                                                             const unsigned int      i_mask_reg_number,
                                                             const unsigned int      i_mask_rnd_exp_cntl,
                                                             const unsigned int      i_imm8 );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_2reg( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_vec_instr,
                                                   const char              i_vector_name,
                                                   const unsigned int      i_gp_reg_base,
                                                   const unsigned int      i_gp_reg_idx,
                                                   const unsigned int      i_scale,
                                                   const int               i_displacement,
                                                   const unsigned int      i_use_broadcast,
                                                   const unsigned int      i_reg_number_src1,
                                                   const unsigned int      i_reg_number_dst );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_2reg_mask( libxsmm_generated_code* io_generated_code,
                                                        const unsigned int      i_vec_instr,
                                                        const char              i_vector_name,
                                                        const unsigned int      i_gp_reg_base,
                                                        const unsigned int      i_gp_reg_idx,
                                                        const unsigned int      i_scale,
                                                        const int               i_displacement,
                                                        const unsigned int      i_use_broadcast,
                                                        const unsigned int      i_reg_number_src1,
                                                        const unsigned int      i_reg_number_dst,
                                                        const unsigned int      i_mask_reg_number,
                                                        const unsigned int      i_mask_rnd_exp_cntl );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_2reg_imm8( libxsmm_generated_code* io_generated_code,
                                                        const unsigned int      i_vec_instr,
                                                        const char              i_vector_name,
                                                        const unsigned int      i_gp_reg_base,
                                                        const unsigned int      i_gp_reg_idx,
                                                        const unsigned int      i_scale,
                                                        const int               i_displacement,
                                                        const unsigned int      i_use_broadcast,
                                                        const unsigned int      i_reg_number_src1,
                                                        const unsigned int      i_reg_number_dst,
                                                        const unsigned int      i_imm8 );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_1reg_mask_imm8( libxsmm_generated_code* io_generated_code,
                                                             const unsigned int      i_vec_instr,
                                                             const char              i_vector_name,
                                                             const unsigned int      i_gp_reg_base,
                                                             const unsigned int      i_gp_reg_idx,
                                                             const unsigned int      i_scale,
                                                             const int               i_displacement,
                                                             const unsigned int      i_use_broadcast,
                                                             const unsigned int      i_reg_number_dst,
                                                             const unsigned int      i_mask_reg_number,
                                                             const unsigned int      i_mask_rnd_exp_cntl,
                                                             const unsigned int      i_imm8 );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_1reg( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_vec_instr,
                                                   const char              i_vector_name,
                                                   const unsigned int      i_gp_reg_base,
                                                   const unsigned int      i_gp_reg_idx,
                                                   const unsigned int      i_scale,
                                                   const int               i_displacement,
                                                   const unsigned int      i_use_broadcast,
                                                   const unsigned int      i_reg_number_dst );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_1reg_mask( libxsmm_generated_code* io_generated_code,
                                                        const unsigned int      i_vec_instr,
                                                        const char              i_vector_name,
                                                        const unsigned int      i_gp_reg_base,
                                                        const unsigned int      i_gp_reg_idx,
                                                        const unsigned int      i_scale,
                                                        const int               i_displacement,
                                                        const unsigned int      i_use_broadcast,
                                                        const unsigned int      i_reg_number_dst,
                                                        const unsigned int      i_mask_reg_number,
                                                        const unsigned int      i_mask_rnd_exp_cntl );

LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_1reg_imm8( libxsmm_generated_code* io_generated_code,
                                                        const unsigned int      i_vec_instr,
                                                        const char              i_vector_name,
                                                        const unsigned int      i_gp_reg_base,
                                                        const unsigned int      i_gp_reg_idx,
                                                        const unsigned int      i_scale,
                                                        const int               i_displacement,
                                                        const unsigned int      i_use_broadcast,
                                                        const unsigned int      i_reg_number_dst,
                                                        const unsigned int      i_imm8 );


LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vex_evex_mask_mov( libxsmm_generated_code* io_generated_code,
                                                const unsigned int      i_vmove_instr,
                                                const unsigned int      i_gp_reg_base,
                                                const unsigned int      i_reg_idx,
                                                const unsigned int      i_scale,
                                                const int               i_displacement,
                                                const char              i_vector_name,
                                                const unsigned int      i_vec_reg_number_0,
                                                const unsigned int      i_use_masking,
                                                const unsigned int      i_mask_reg_number,
                                                const unsigned int      i_is_store );

/* TODO: check if we can merge this alu_imm */
/**
 * Generates prefetch instructions with displacements, SIB addressing is not
 * supported by this function
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_prefetch_instr actual prefetch variant
 * @param i_gp_reg_base the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_displacement the offset to the base address
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_prefetch( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_prefetch_instr,
                                       const unsigned int      i_gp_reg_base,
                                       const unsigned int      i_gp_reg_idx,
                                       const unsigned int      i_scale,
                                       const int               i_displacement );

/**
 * Generates alu memory movements like movq 7(%rax,%rbx,2), %rcx
 * Takes 3 gp_registers (0-15 values)
 * i_is_store tells whether this is a store or load
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_alu_mem( libxsmm_generated_code* io_generated_code,
                                      const unsigned int     i_alu_instr,
                                      const unsigned int     i_gp_reg_base,
                                      const unsigned int     i_gp_reg_idx,
                                      const unsigned int     i_scale,
                                      const int              i_displacement,
                                      const unsigned int     i_gp_reg_number,
                                      const unsigned int     i_is_store );

/**
 * Generates regular all instructions with immediates
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_alu_instr actual alu gpr instruction
 * @param i_gp_reg_number the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_immediate the immediate operand
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_alu_imm( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_alu_instr,
                                      const unsigned int      i_gp_reg_number,
                                      const long long         i_immediate );

/**
 * Generates regular all instructions with immediates, 64bit
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_alu_instr actual alu gpr instruction
 * @param i_gp_reg_number the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_immediate the immediate operand
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_alu_imm_i64( libxsmm_generated_code* io_generated_code,
                                          const unsigned int      i_alu_instr,
                                          const unsigned int      i_gp_reg_number,
                                          const long long         i_immediate );

/**
 * Generates regular all instructions with immediates
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_alu_instr actual alu gpr instruction
 * @param i_gp_reg_number_src the source register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_gp_reg_number_dest the destination register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_alu_reg( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_alu_instr,
                                      const unsigned int      i_gp_reg_number_src,
                                      const unsigned int      i_gp_reg_number_dest);

/**
 * Generates push to the stack for a GPR
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gp_reg_number the source register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_push_reg( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_gp_reg_number );

/**
 * Generates pop from the stack for a GPR
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gp_reg_number the source register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_pop_reg( libxsmm_generated_code* io_generated_code,
                                      const unsigned int      i_gp_reg_number );

/**
 * Allows for mask move instructions in AVX512
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_mask_instr actual mask move instruction
 * @param i_gp_reg_number the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_mask_reg_number the register number (k1=1...k7=7)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_mask_move( libxsmm_generated_code* io_generated_code,
                                        const unsigned int      i_mask_instr,
                                        const unsigned int      i_gp_reg_number,
                                        const unsigned int      i_mask_reg_number );

/**
 * Allows for mask move instructions in AVX512
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_mask_instr actual mask move instruction
 * @param i_gp_reg_base base address register for memory broadcast
 * @param i_gp_reg_idx index register for memory broadcast, can be LIBXSMM_X86_GP_REG_UNDEF -> then regular displacement version is generated
 * @param i_scale scale of index register, ignored if i_gp_reg_idx is LIBXSMM_X86_GP_REG_UNDEF
 * @param i_displacement displacement to SIB address
 * @param i_mask_reg_number the register number (k1=1...k7=7)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_mask_move_mem( libxsmm_generated_code* io_generated_code,
                                            const unsigned int      i_mask_instr,
                                            const unsigned int      i_gp_reg_base,
                                            const unsigned int      i_gp_reg_idx,
                                            const unsigned int      i_scale,
                                            const int               i_displacement,
                                            const unsigned int      i_mask_reg_number );

/**
 * Allows for mask move instructions in AVX512
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_mask_instr actual mask compute instruction
 * @param i_mask_reg_number_src_0 the first operand register number (att syntax) (k1=1...k7=7)
 * @param i_mask_reg_number_src_1 the second operand register number (att syntax) (k1=1...k7=7)
 * @param i_mask_reg_number_dest the third operand register number (att syntax) (k1=1...k7=7)
 * @param i_imm8 immediate value
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_mask_compute_reg( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_mask_instr,
                                               const unsigned int      i_mask_reg_number_src_0,
                                               const unsigned int      i_mask_reg_number_src_1,
                                               const unsigned int      i_mask_reg_number_dest,
                                               const unsigned int      i_imm8 );

/**
 * Generates a label to which one can jump back and pushes it on the loop label stack
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param io_loop_label_tracker data structure to handle loop labels, nested loops are supported, but not overlapping loops
*/
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_register_jump_back_label( libxsmm_generated_code*     io_generated_code,
                                                       libxsmm_loop_label_tracker* io_loop_label_tracker );

/**
 * Pops the latest from the loop label stack and jumps there
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_jmp_instr the particular jump instruction used
 * @param io_loop_label_tracker data structure to handle loop labels will jump to latest registered label
*/
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_jump_back_to_label( libxsmm_generated_code*     io_generated_code,
                                                 const unsigned int          i_jmp_instr,
                                                 libxsmm_loop_label_tracker* io_loop_label_tracker );

/**
 * Generates a label to which one can jump back and pushes it on the loop label stack
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_label_no position in the jump label tracker to set
 * @param io_jump_label_tracker forward jump tracker structure for tracking the jump addresses/labels
*/
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_register_jump_label( libxsmm_generated_code*     io_generated_code,
                                                  const unsigned int          i_label_no,
                                                  libxsmm_jump_label_tracker* io_jump_label_tracker );

/**
 * Jumps to the address/label stored a specific position
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_jmp_instr the particular jump instruction used
 * @param i_label_no position in the jump label tracker to jump to
 * @param io_jump_label_tracker data structures that tracks arbitrary jump labels
*/
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_jump_to_label( libxsmm_generated_code*     io_generated_code,
                                            const unsigned int          i_jmp_instr,
                                            const unsigned int          i_label_no,
                                            libxsmm_jump_label_tracker* io_jump_label_tracker );

/**
 * Generates an insertion of constants into the code stream and loads them into
 * into a vector register
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_data pointer to an array of bytes that should be loaded, length needs to match registerlength specified in i_vector_name (x=16, y=32, z=64)
 * @param i_id global identifier of constants to load.
 * @param i_vector_name the vector register name prefix (x,y or z)
 * @param i_vec_reg_number the destination(gather)/source(scatter) vec register (xmm/ymm: 0-15, zmm: 0-31)
*/
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_full_vec_load_of_constants ( libxsmm_generated_code *io_generated_code,
                                                          const unsigned char *i_data,
                                                          const char *i_id,
                                                          const char i_vector_name,
                                                          const unsigned int i_vec_reg_number );

/**
 * Executes rdseed, checks carry, retries, and resets flags
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gp_reg_number the destination of the rdseed
*/
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_rdseed_load ( libxsmm_generated_code *io_generated_code,
                                           const unsigned int      i_gp_reg_number );

/**
 * Generates ld/stconfig/tilerelease instructions
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_id id of AMX region
 * @param i_instruction_set requested instruction set to encode
 * @param i_tcontrol_instr actual tile mem instruction variant
 * @param i_gp_reg_base base register which address where to store/load tile config
 * @param i_displacement displacement to i_gp_reg_base
 * @param i_tile_config pointer to tile config structure
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_tile_control( libxsmm_generated_code*    io_generated_code,
                                           const unsigned int         i_id,
                                           const unsigned int         i_instruction_set,
                                           const unsigned int         i_tcontrol_instr,
                                           const unsigned int         i_gp_reg_base,
                                           const int                  i_displacement,
                                           const libxsmm_tile_config* i_tile_config );

/**
 * Generates tilemove/tilestore instructions
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_instruction_set requested instruction set to encode
 * @param i_tmove_instr actual tile mem instruction variant
 * @param i_gp_reg_base the base register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_gp_reg_idx the base register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_scale scaling factor of idx
 * @param i_displacement the offset to the base address
 * @param i_tile_reg_number the tile register number (tmm: 0-7)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_tile_move( libxsmm_generated_code* io_generated_code,
                                        const unsigned int      i_instruction_set,
                                        const unsigned int      i_tmove_instr,
                                        const unsigned int      i_gp_reg_base,
                                        const unsigned int      i_gp_reg_idx,
                                        const unsigned int      i_scale,
                                        const int               i_displacement,
                                        const unsigned int      i_tile_reg_number );

/**
 * Generates tilecompute instructions
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_instruction_set requested instruction set to encode
 * @param i_tcompute_instr actual tile compute instruction variant
 * @param i_tile_src_reg_number_0 the 1st src tile register number (tmm: 0-7)
 * @param i_tile_src_reg_number_1 the 2nd src tile register number (tmm: 0-7) (might be ignored by some instuctions)
 * @param i_tile_dst_reg_number the dst tile register number (tmm: 0-7)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_tile_compute( libxsmm_generated_code* io_generated_code,
                                           const unsigned int      i_instruction_set,
                                           const unsigned int      i_tcompute_instr,
                                           const unsigned int      i_tile_src_reg_number_0,
                                           const unsigned int      i_tile_src_reg_number_1,
                                           const unsigned int      i_tile_dst_reg_number );

#endif /* GENERATOR_X86_INSTRUCTIONS_H */
