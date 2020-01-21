/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Greg Henry (Intel Corp.)
******************************************************************************/

#ifndef GENERATOR_X86_INSTRUCTIONS_H
#define GENERATOR_X86_INSTRUCTIONS_H

#include "generator_common.h"

/**
 * Opens the inline assembly section / jit stream
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gp_reg_mapping gp register mapping for initialization
 * @param i_prefetch prefetch mode which may result in additional gp reg inits
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream( libxsmm_generated_code*       io_generated_code,
                                          const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                          unsigned int                  i_prefetch );

/**
 * Closes the inline assembly section / jit stream
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gp_reg_mapping gp register mapping for clobbering
 * @param i_prefetch prefetch mode which may result in additional gp reg clobbers
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream( libxsmm_generated_code*       io_generated_code,
                                           const libxsmm_gp_reg_mapping* i_gp_reg_mapping,
                                           unsigned int                  i_prefetch );

/**
 * Generates vmaskmovps/vmaskmovpd with displacements for loads and stores.
 * Only works with i_vector_name='Y'
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_mask_move( libxsmm_generated_code* io_generated_code,
                                     const unsigned int      i_vmove_instr,
                                     const unsigned int      i_gp_reg_base,
                                     const unsigned int      i_gp_reg_idx,
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
 * @param i_gp_reg_number the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_displacement the offset to the base address
 * @param i_vector_name the vector register name prefix (x, y or z)
 * @param i_vec_reg_number_0 the vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_mask_reg_number the mask register to be used
 * @param i_use_zero_masking: 0: merge masking ; !=0: zero masking
 * @param i_is_store 0: load semantic, other: store semantic
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_move( libxsmm_generated_code* io_generated_code,
                                       const unsigned int      i_instruction_set,
                                       const unsigned int      i_vmove_instr,
                                       const unsigned int      i_gp_reg_base,
                                       const unsigned int      i_gp_reg_idx,
                                       const unsigned int      i_scale,
                                       const int               i_displacement,
                                       const char              i_vector_name,
                                       const unsigned int      i_vec_reg_number_0,
                                       const unsigned int      i_mask_reg_number,
                                       const unsigned int      i_use_zero_masking,
                                       const unsigned int      i_is_store );

/**
 * Generates (v)XYZpd/(v)XYZps/(v)XYZsd/(v)XYZss instructions with 2 or 3 vector registers, memory operands are not supported as first operand
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_instruction_set requested instruction set to encode
 * @param i_vec_instr actual operation variant
 * @param i_vector_name the vector register name prefix (x,y or z)
 * @param i_vec_reg_number_0 the first vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_vec_reg_number_1 the second vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_vec_reg_number_2 the third vector register number (xmm/ymm: 0-15, zmm: 0-31), if this operand equals LIBXSMM_X86_VEC_REG_UNDEF -> SSE3 code generation
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_reg( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_instruction_set,
                                              const unsigned int      i_vec_instr,
                                              const char              i_vector_name,
                                              const unsigned int      i_vec_reg_number_0,
                                              const unsigned int      i_vec_reg_number_1,
                                              const unsigned int      i_vec_reg_number_2 );


/**
 * Generates (v)XYZpd/(v)XYZps/(v)XYZsd/(v)XYZss convert instructions with 2 vector registers, memory operands are not supported as first operand
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_instruction_set requested instruction set to encode
 * @param i_vec_instr actual operation variant
 * @param i_vector_name the vector register name prefix (x,y or z)
 * @param i_vec_reg_src_0 the first source vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_vec_reg_src_1 the second source vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_vec_reg_dst the destination vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_shuffle_operand is an immediate (only looked at when needed)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_convert ( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_instruction_set,
                                              const unsigned int      i_vec_instr,
                                              const char              i_vector_name,
                                              const unsigned int      i_vec_reg_src_0,
                                              const unsigned int      i_vec_reg_src_1,
                                              const unsigned int      i_vec_reg_dst,
                                              const unsigned int      i_shuffle_operand );

/**
 * Generates (v)XYZpd/(v)XYZps/(v)XYZsd/(v)XYZss instructions with 3 vector registers and masking
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_instruction_set requested instruction set to encode
 * @param i_vec_instr actual operation variant
 * @param i_vector_name the vector register name prefix (z)
 * @param i_vec_reg_number_0 the first vector register number (zmm: 0-31)
 * @param i_vec_reg_number_1 the second vector register number (zmm: 0-31)
 * @param i_vec_reg_number_3 the second vector register number (zmm: 0-31)
 * @param i_immediate immediate just as the compare value for a compare instruction
 * @param i_mask_reg_number the mask register to read/write
 * @param i_use_zero_masking 0: merge masking, !=0 zero masking
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_reg_mask( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_instruction_set,
                                              const unsigned int      i_vec_instr,
                                              const char              i_vector_name,
                                              const unsigned int      i_vec_reg_number_0,
                                              const unsigned int      i_vec_reg_number_1,
                                              const unsigned int      i_vec_reg_number_2,
                                              const unsigned int      i_immediate,
                                              const unsigned int      i_mask_reg_number,
                                              const unsigned int      i_use_zero_masking );

/**
 * @param i_instruction_set requested instruction set to encode
 * @param i_vec_instr actual operation variant
 * @param i_vector_name the vector register name prefix (z)
 * @param i_vec_reg_number_0 the first vector register number (zmm: 0-31)
 * @param i_vec_reg_number_1 the second vector register number (zmm: 0-31)
 * @param i_vec_reg_number_2 the third vector register number (zmm: 0-31)
 * @param i_mask_reg_number the mask register (0-7)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_instruction_set,
                                              const unsigned int      i_vec_instr,
                                              const unsigned int      i_use_broadcast,
                                              const unsigned int      i_gp_reg_base,
                                              const unsigned int      i_gp_reg_idx,
                                              const unsigned int      i_scale,
                                              const int               i_displacement,
                                              const char              i_vector_name,
                                              const unsigned int      i_vec_reg_number_0,
                                              const unsigned int      i_vec_reg_number_1 );

/**
 * Generates vector instructions which require an immediate and mask. immediate is optional.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_instruction_set requested instruction set to encode
 * @param i_vec_instr actual operation variant
 * @param i_use_broadcast if != 0 memory operand is interpreted as a scalar and broadcasted in fused fashion, only supported on AVX512
 * @param i_gp_reg_base base address register for memory broadcast
 * @param i_gp_reg_idx index register for memory broadcast, can be LIBXSMM_X86_GP_REG_UNDEF -> then regular displacement version is generated
 * @param i_scale scale of index register, ignored if i_gp_reg_idx is LIBXSMM_X86_GP_REG_UNDEF
 * @param i_displacement displacement to SIB address
 * @param i_vector_name the vector register name prefix (z)
 * @param i_vec_reg_number_0 the first vector register number (zmm: 0-31)
 * @param i_vec_reg_number_1 the second vector register number (zmm: 0-31)
 * @param i_immediate immediate just as the compare value for a compare instruction
 * @param i_mask_reg_number the mask register to read/write
 * @param i_use_zero_masking 0: merge masking; !=0: zero masking
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_mem_mask( libxsmm_generated_code* io_generated_code,
                                                   const unsigned int      i_instruction_set,
                                                   const unsigned int      i_vec_instr,
                                                   const unsigned int      i_use_broadcast,
                                                   const unsigned int      i_gp_reg_base,
                                                   const unsigned int      i_gp_reg_idx,
                                                   const unsigned int      i_scale,
                                                   const int               i_displacement,
                                                   const char              i_vector_name,
                                                   const unsigned int      i_vec_reg_number_0,
                                                   const unsigned int      i_vec_reg_number_1,
                                                   const unsigned int      i_immediate,
                                                   const unsigned int      i_mask_reg_number,
                                                   const unsigned int      i_use_zero_masking );

 /**
  * Generates quadmadd instructions added in Knights Mill
  *
  * @param io_generated_code pointer to the pointer of the generated code structure
  * @param i_instruction_set requested instruction set to encode
  * @param i_vec_instr actual operation variant
  * @param i_gp_reg_base base address register for memory broadcast
  * @param i_gp_reg_idx index register for memory broadcast, can be LIBXSMM_X86_GP_REG_UNDEF -> then regular displacement version is generated
  * @param i_scale scale of index register, ignored if i_gp_reg_idx is LIBXSMM_X86_GP_REG_UNDEF
  * @param i_displacement displacement to SIB address
  * @param i_vector_name the vector register name prefix (z)
  * @param i_vec_reg_number_src the second vector register number (zmm: 0-31), this define a implicit register range
  * @param i_vec_reg_number_dest the first vector register number (zmm: 0-31)
  */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_compute_qfma( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_instruction_set,
                                               const unsigned int      i_vec_instr,
                                               const unsigned int      i_gp_reg_base,
                                               const unsigned int      i_gp_reg_idx,
                                               const unsigned int      i_scale,
                                               const int               i_displacement,
                                               const char              i_vector_name,
                                               const unsigned int      i_vec_reg_number_src,
                                               const unsigned int      i_vec_reg_number_dest );

/**
 * Generates shuffle instructions with 2 or 3 vector registers, memory operands are not supported as first operand
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_instruction_set requested instruction set to encode
 * @param i_vec_instr actual operation variant
 * @param i_vector_name the vector register name prefix (x,y or z)
 * @param i_vec_reg_number_0 the first vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_vec_reg_number_1 the second vector register number (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_vec_reg_number_2 the third vector register number (xmm/ymm: 0-15, zmm: 0-31), if this operand equals LIBXSMM_X86_VEC_REG_UNDEF -> SSE3 code generation
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_shuffle_reg( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_instruction_set,
                                              const unsigned int      i_vec_instr,
                                              const char              i_vector_name,
                                              const unsigned int      i_vec_reg_number_0,
                                              const unsigned int      i_vec_reg_number_1,
                                              const unsigned int      i_vec_reg_number_2,
                                              const unsigned int      i_shuffle_operand );

/**
 * Generates shuffle instructions with 2 or 3 vector registers, memory operands are not supported as first operand
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_instruction_set requested instruction set to encode
 * @param i_vmove_instr actual operation variant (gather/scatter and single/double)
 * @param i_vector_name the vector register name prefix (x,y or z)
 * @param i_gp_reg_number the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
 * @param i_vec_reg_idx the index vector registers (ymm0-15 AVX2) (zmm0-zmm32 AVX512)
 * @param i_scale the scaling of the indexes in i_vec_reg_idx
 * @param i_displacement the offset to the base address
 * @param i_vec_reg_number the destination(gather)/source(scatter) vec register (xmm/ymm: 0-15, zmm: 0-31)
 * @param i_mask_reg_number the mask register (xmm/ymm: 0-15 when using AVX2), (k1-k7 when using AVX512)
 * @param i_is_gather "true" generate a gather instruction, "false" generator a scatter instruction
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_vec_move_gathscat( libxsmm_generated_code* io_generated_code,
                                                const unsigned int      i_instruction_set,
                                                const unsigned int      i_vmove_instr,
                                                const char              i_vector_name,
                                                const unsigned int      i_gp_reg_base,
                                                const unsigned int      i_vec_reg_idx,
                                                const unsigned int      i_scale,
                                                const int               i_displacement,
                                                const unsigned int      i_vec_reg_number,
                                                const unsigned int      i_mask_reg_number,
                                                const unsigned int      i_is_gather );

/* @TODO check if we can merge this alu_imm */
/**
 * Generates prefetch instructions with displacements, SIB addressing is not
 * supported by this function
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_prefetch_instr actual prefetch variant
 * @param i_gp_reg_number the register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15) of the base address register
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
                                          const size_t            i_immediate );

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
 * @param i_mask_instr actual mask compute instruction
 * @param i_mask_reg_number_src_0 the first operand register number (att syntax) (k1=1...k7=7)
 * @param i_mask_reg_number_src_1 the second operand register number (att syntax) (k1=1...k7=7)
 * @param i_mask_reg_number_dest the third operand register number (att syntax) (k1=1...k7=7)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_mask_compute_reg( libxsmm_generated_code* io_generated_code,
                                               const unsigned int      i_mask_instr,
                                               const unsigned int      i_mask_reg_number_src_0,
                                               const unsigned int      i_mask_reg_number_src_1,
                                               const unsigned int      i_mask_reg_number_dest  );

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
 * @parma i_labal_no position in the jump label tracker to set
 * @param io_jump_forward_label_tracker forward jump tracker structure for tracking the jump addresses/labels
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
 * Generates a sequence to load function arguments from the stack (arguments )
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_arg_number the number of an argument which was passed on the stack
 * @param i_gp_reg_number the destination register number (rax=0,rcx=1,rdx=2,rbx=3,rsp=4,rbp=5,rsi=6,rdi=7,r8=8,r9=9,r10=10,r11=11,r12=12,r13=13,r14=14,r15=15)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_load_arg_to_reg( libxsmm_generated_code* io_generated_code,
                                              const unsigned int      i_arg_number,
                                              const unsigned int      i_gp_reg_number );

/**
 * @TODO: clean-up
 * Opens the inline assembly section / jit stream for matcopy, this is hacked and should be cleaned up
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_arch architecture code was generated for (needed to build clobber)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream_matcopy( libxsmm_generated_code*                   io_generated_code,
                                                  const unsigned int                        i_gp_reg_a,
                                                  const unsigned int                        i_gp_reg_lda,
                                                  const unsigned int                        i_gp_reg_b,
                                                  const unsigned int                        i_gp_reg_ldb,
                                                  const unsigned int                        i_gp_reg_a_pf,
                                                  const unsigned int                        i_gp_reg_b_pf,
                                                  const char*                               i_arch );

/**
 * @TODO: clean-up
 * Closes the inline assembly section / jit stream for matcopy, this is hacked and should be cleaned up
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_arch architecture code was generated for (needed to build clobber)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream_matcopy( libxsmm_generated_code*       io_generated_code,
                                                   const char*                   i_arch );

/**
 * @TODO: clean-up
 * Opens the inline assembly section / jit stream for transposes, this is hacked and should be cleaned up
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_arch architecture code was generated for (needed to build clobber)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_open_stream_transpose( libxsmm_generated_code*                   io_generated_code,
                                                    const unsigned int                        i_gp_reg_a,
                                                    const unsigned int                        i_gp_reg_lda,
                                                    const unsigned int                        i_gp_reg_b,
                                                    const unsigned int                        i_gp_reg_ldb,
                                                    const char*                               i_arch );

/**
 * @TODO: clean-up
 * Closes the inline assembly section / jit stream for transposes, this is hacked and should be cleaned up
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_arch architecture code was generated for (needed to build clobber)
 */
LIBXSMM_API_INTERN
void libxsmm_x86_instruction_close_stream_transpose( libxsmm_generated_code*       io_generated_code,
                                                     const char*                   i_arch );

#endif /* GENERATOR_X86_INSTRUCTIONS_H */

