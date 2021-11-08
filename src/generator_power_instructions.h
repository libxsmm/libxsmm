/******************************************************************************
* Copyright (c) Friedrich Schiller University Jena - All rights reserved.     *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Breuer (Univ. Jena)
******************************************************************************/

#ifndef GENERATOR_POWER_INSTRUCTIONS_H
#define GENERATOR_POWER_INSTRUCTIONS_H

#include "generator_common.h"
#include "../include/libxsmm_typedefs.h"

/* general purpose registers */
#define LIBXSMM_POWER_GPR_R0   0
#define LIBXSMM_POWER_GPR_R1   1
#define LIBXSMM_POWER_GPR_R2   2
#define LIBXSMM_POWER_GPR_R3   3
#define LIBXSMM_POWER_GPR_R4   4
#define LIBXSMM_POWER_GPR_R5   5
#define LIBXSMM_POWER_GPR_R6   6
#define LIBXSMM_POWER_GPR_R7   7
#define LIBXSMM_POWER_GPR_R8   8
#define LIBXSMM_POWER_GPR_R9   9
#define LIBXSMM_POWER_GPR_R10 10
#define LIBXSMM_POWER_GPR_R11 11
#define LIBXSMM_POWER_GPR_R12 12
#define LIBXSMM_POWER_GPR_R13 13
#define LIBXSMM_POWER_GPR_R14 14
#define LIBXSMM_POWER_GPR_R15 15
#define LIBXSMM_POWER_GPR_R16 16
#define LIBXSMM_POWER_GPR_R17 17
#define LIBXSMM_POWER_GPR_R18 18
#define LIBXSMM_POWER_GPR_R19 19
#define LIBXSMM_POWER_GPR_R20 20
#define LIBXSMM_POWER_GPR_R21 21
#define LIBXSMM_POWER_GPR_R22 22
#define LIBXSMM_POWER_GPR_R23 23
#define LIBXSMM_POWER_GPR_R24 24
#define LIBXSMM_POWER_GPR_R25 25
#define LIBXSMM_POWER_GPR_R26 26
#define LIBXSMM_POWER_GPR_R27 27
#define LIBXSMM_POWER_GPR_R28 28
#define LIBXSMM_POWER_GPR_R29 29
#define LIBXSMM_POWER_GPR_R30 30
#define LIBXSMM_POWER_GPR_R31 31

/* special registers */
#define LIBXSMM_POWER_GPR_SP 1
#define LIBXSMM_POWER_SPR_CTR 288 /* reversed 5-bit parts: 01001 00000 */

/* floating-point registers */
#define LIBXSMM_POWER_FPR_F0   0
#define LIBXSMM_POWER_FPR_F1   1
#define LIBXSMM_POWER_FPR_F2   2
#define LIBXSMM_POWER_FPR_F3   3
#define LIBXSMM_POWER_FPR_F4   4
#define LIBXSMM_POWER_FPR_F5   5
#define LIBXSMM_POWER_FPR_F6   6
#define LIBXSMM_POWER_FPR_F7   7
#define LIBXSMM_POWER_FPR_F8   8
#define LIBXSMM_POWER_FPR_F9   9
#define LIBXSMM_POWER_FPR_F10 10
#define LIBXSMM_POWER_FPR_F11 11
#define LIBXSMM_POWER_FPR_F12 12
#define LIBXSMM_POWER_FPR_F13 13
#define LIBXSMM_POWER_FPR_F14 14
#define LIBXSMM_POWER_FPR_F15 15
#define LIBXSMM_POWER_FPR_F16 16
#define LIBXSMM_POWER_FPR_F17 17
#define LIBXSMM_POWER_FPR_F18 18
#define LIBXSMM_POWER_FPR_F19 19
#define LIBXSMM_POWER_FPR_F20 20
#define LIBXSMM_POWER_FPR_F21 21
#define LIBXSMM_POWER_FPR_F22 22
#define LIBXSMM_POWER_FPR_F23 23
#define LIBXSMM_POWER_FPR_F24 24
#define LIBXSMM_POWER_FPR_F25 25
#define LIBXSMM_POWER_FPR_F26 26
#define LIBXSMM_POWER_FPR_F27 27
#define LIBXSMM_POWER_FPR_F28 28
#define LIBXSMM_POWER_FPR_F29 29
#define LIBXSMM_POWER_FPR_F30 30
#define LIBXSMM_POWER_FPR_F31 31

/* vector status and control register */
#define LIBXSMM_POWER_VSR_VS0   0
#define LIBXSMM_POWER_VSR_VS1   1
#define LIBXSMM_POWER_VSR_VS2   2
#define LIBXSMM_POWER_VSR_VS3   3
#define LIBXSMM_POWER_VSR_VS4   4
#define LIBXSMM_POWER_VSR_VS5   5
#define LIBXSMM_POWER_VSR_VS6   6
#define LIBXSMM_POWER_VSR_VS7   7
#define LIBXSMM_POWER_VSR_VS8   8
#define LIBXSMM_POWER_VSR_VS9   9
#define LIBXSMM_POWER_VSR_VS10 10
#define LIBXSMM_POWER_VSR_VS11 11
#define LIBXSMM_POWER_VSR_VS12 12
#define LIBXSMM_POWER_VSR_VS13 13
#define LIBXSMM_POWER_VSR_VS14 14
#define LIBXSMM_POWER_VSR_VS15 15
#define LIBXSMM_POWER_VSR_VS16 16
#define LIBXSMM_POWER_VSR_VS17 17
#define LIBXSMM_POWER_VSR_VS18 18
#define LIBXSMM_POWER_VSR_VS19 19
#define LIBXSMM_POWER_VSR_VS20 20
#define LIBXSMM_POWER_VSR_VS21 21
#define LIBXSMM_POWER_VSR_VS22 22
#define LIBXSMM_POWER_VSR_VS23 23
#define LIBXSMM_POWER_VSR_VS24 24
#define LIBXSMM_POWER_VSR_VS25 25
#define LIBXSMM_POWER_VSR_VS26 26
#define LIBXSMM_POWER_VSR_VS27 27
#define LIBXSMM_POWER_VSR_VS28 28
#define LIBXSMM_POWER_VSR_VS29 29
#define LIBXSMM_POWER_VSR_VS30 30
#define LIBXSMM_POWER_VSR_VS31 31
#define LIBXSMM_POWER_VSR_VS32 32
#define LIBXSMM_POWER_VSR_VS33 33
#define LIBXSMM_POWER_VSR_VS34 34
#define LIBXSMM_POWER_VSR_VS35 35
#define LIBXSMM_POWER_VSR_VS36 36
#define LIBXSMM_POWER_VSR_VS37 37
#define LIBXSMM_POWER_VSR_VS38 38
#define LIBXSMM_POWER_VSR_VS39 39
#define LIBXSMM_POWER_VSR_VS40 40
#define LIBXSMM_POWER_VSR_VS41 41
#define LIBXSMM_POWER_VSR_VS42 42
#define LIBXSMM_POWER_VSR_VS43 43
#define LIBXSMM_POWER_VSR_VS44 44
#define LIBXSMM_POWER_VSR_VS45 45
#define LIBXSMM_POWER_VSR_VS46 46
#define LIBXSMM_POWER_VSR_VS47 47
#define LIBXSMM_POWER_VSR_VS48 48
#define LIBXSMM_POWER_VSR_VS49 49
#define LIBXSMM_POWER_VSR_VS50 50
#define LIBXSMM_POWER_VSR_VS51 51
#define LIBXSMM_POWER_VSR_VS52 52
#define LIBXSMM_POWER_VSR_VS53 53
#define LIBXSMM_POWER_VSR_VS54 54
#define LIBXSMM_POWER_VSR_VS55 55
#define LIBXSMM_POWER_VSR_VS56 56
#define LIBXSMM_POWER_VSR_VS57 57
#define LIBXSMM_POWER_VSR_VS58 58
#define LIBXSMM_POWER_VSR_VS59 59
#define LIBXSMM_POWER_VSR_VS60 60
#define LIBXSMM_POWER_VSR_VS61 61
#define LIBXSMM_POWER_VSR_VS62 62
#define LIBXSMM_POWER_VSR_VS63 63

/* accumulators */
#define LIBXSMM_POWER_ACC_A0 0
#define LIBXSMM_POWER_ACC_A1 1
#define LIBXSMM_POWER_ACC_A2 2
#define LIBXSMM_POWER_ACC_A3 3
#define LIBXSMM_POWER_ACC_A4 4
#define LIBXSMM_POWER_ACC_A5 5
#define LIBXSMM_POWER_ACC_A6 6
#define LIBXSMM_POWER_ACC_A7 7

/* undefined instruction */
#define LIBXSMM_POWER_INSTR_UNDEF 9999

/* branch facility */
#define LIBXSMM_POWER_INSTR_B_BC 0x40000000

/* fixed-point storage access */
#define LIBXSMM_POWER_INSTR_FIP_LD 0xe8000000
#define LIBXSMM_POWER_INSTR_FIP_STD 0xf8000000

/* fixed-point arithmetic */
#define LIBXSMM_POWER_INSTR_FIP_ADDI 0x38000000

/* fixed-point compare */
#define LIBXSMM_POWER_INSTR_FIP_CMPI 0x2c000000

/* fixed-point logical */
#define LIBXSMM_POWER_INSTR_FIP_ORI  0x60000000

/* fixed-point rotate and shift */
#define LIBXSMM_POWER_INSTR_FIP_RLDICR 0x78000004

/* fixed-point move to/from system regisster */
#define LIBXSMM_POWER_INSTR_FIP_MTSPR 0x7c0003a6

/* floating-point storage access */
#define LIBXSMM_POWER_INSTR_FLP_LFD 0xc8000000
#define LIBXSMM_POWER_INSTR_FLP_STFD 0xd8000000

/* vector storage access */
#define LIBXSMM_POWER_INSTR_VEC_LVX 0x7c0000ce
#define LIBXSMM_POWER_INSTR_VEC_STVX 0x7c0001ce

/* vector-scalar extension storage access */
#define LIBXSMM_POWER_INSTR_VSX_LXVW4X  0x7c000618
#define LIBXSMM_POWER_INSTR_VSX_STXVW4X 0x7c000718
#define LIBXSMM_POWER_INSTR_VSX_LXVWSX 0x7c0002d8
#define LIBXSMM_POWER_INSTR_VSX_LXVLL 0x7c00025a
#define LIBXSMM_POWER_INSTR_VSX_STXVLL 0x7c00035a

/* vector-scalar extension compute */
#define LIBXSMM_POWER_INSTR_VSX_XVMADDASP 0xf0000208

/* vector-scalar extension permute */
#define LIBXSMM_POWER_INSTR_VSX_XXBRD 0xf017076c
#define LIBXSMM_POWER_INSTR_VSX_XXBRW 0xf00f076c

/**
 * Generates a conditional branch instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_bo conditions under which the branch is taken.
 * @param i_bi condition register bit (bi+32).
 * @param i_bd 14-bit immediate relative or absolute branch address.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_b_conditional( unsigned int  i_instr,
                                                      unsigned char i_bo,
                                                      unsigned char i_bi,
                                                      unsigned int  i_bd );

/**
 * Generates a fixed-point storage access instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_rs destination (load) or source (store) GPR.
 * @param i_ra source GPR holding the address.
 * @param i_d 16-bit immediate address offset.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_fip_storage_access( unsigned int  i_instr,
                                                           unsigned char i_rs,
                                                           unsigned char i_ra,
                                                           unsigned int  i_d );

/**
 * Generates an arithmetic scalar fixed-point instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_rt destination GPR.
 * @param i_ra source GPR.
 * @param i_si 16-bit immediate.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_fip_arithmetic( unsigned int  i_instr,
                                                       unsigned char i_rt,
                                                       unsigned char i_ra,
                                                       unsigned int  i_si );

/**
 * Generates compare fixed-point instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_bf destination SI field.
 * @param i_l 0: RA[32:63] are extended to 64 bits, 1: all of RA's bits are used.
 * @param i_ra source register which is compared.
 * @param i_si 16-bit immediate.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_fip_compare( unsigned int  i_instr,
                                                    unsigned char i_bf,
                                                    unsigned char i_l,
                                                    unsigned char i_ra,
                                                    unsigned int  i_si );

/**
 * Generates a logical scalar fixed-point instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_ra destination GPR.
 * @param i_rs source GPR.
 * @param i_ui 16-bit immediate.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_fip_logical( unsigned int  i_instr,
                                                    unsigned char i_ra,
                                                    unsigned char i_rs,
                                                    unsigned int  i_ui );

/**
 * Generates a rotate fixed-point instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_ra destination GPR.
 * @param i_rs source GPR.
 * @param i_sh 6-bit immediate.
 * @param i_mb 6-bit immediate.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_fip_rotate( unsigned int  i_instr,
                                                   unsigned char i_ra,
                                                   unsigned char i_rs,
                                                   unsigned int  i_sh,
                                                   unsigned int  i_mb );

/**
 * Generates a move to/from system register instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_rs source/destination GPR.
 * @param i_spr destination/source SPR.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_fip_system( unsigned int  i_instr,
                                                   unsigned char i_rs,
                                                   unsigned int  i_spr );

/**
 * Generates a floating-point storage access instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_frs destination (load) or source (store) floating-point register.
 * @param i_ra source GPR holding the address.
 * @param i_d 16-bit immediate address offset.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_flp_storage_access( unsigned int  i_instr,
                                                           unsigned char i_frs,
                                                           unsigned char i_ra,
                                                           unsigned int  i_d );

/**
 * Generates a vector storage access instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_vrt destination (load) or source (store) VRT.
 * @param i_ra GPR containing the address offset, offset is 0 if id 0 is given.
 * @param i_rb GPR containing the address.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_vec_storage_access( unsigned int  i_instr,
                                                           unsigned char i_vrt,
                                                           unsigned char i_ra,
                                                           unsigned char i_rb );

/**
 * Generates a VSX storage access instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_xt destination (load) or source (store) VSR.
 * @param i_ra GPR containing the address offset, offset is 0 if id 0 is given.
 * @param i_rb GPR containing the address.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_vsx_storage_access( unsigned int  i_instr,
                                                           unsigned char i_xt,
                                                           unsigned char i_ra,
                                                           unsigned char i_rb );

/**
 * Generates a VSX binary floating-point arithmetic operation performing multiply-add.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_xt destination VSR.
 * @param i_xa first source VSR.
 * @param i_xb second source VSR.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_vsx_vector_bfp_madd( unsigned int  i_instr,
                                                            unsigned char i_xt,
                                                            unsigned char i_xa,
                                                            unsigned char i_xb );

/**
 * Generates a VSX byte-reverse instruction.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_xt destination VSR.
 * @param i_xb source VSR.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_vsx_vector_permute_byte_reverse( unsigned int  i_instr,
                                                                        unsigned char i_xt,
                                                                        unsigned char i_xb );

/**
 * Generates a generic POWER-instruction with two arguments
 * following the syntax of the mnemonics.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_arg0 first argument.
 * @param i_arg1 second argument.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_generic_2( unsigned int i_instr,
                                                  unsigned int i_arg0,
                                                  unsigned int i_arg1 );

/**
 * Generates a generic POWER-instruction with three arguments
 * following the syntax of the mnemonics.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_arg0 first argument.
 * @param i_arg1 second argument.
 * @param i_arg2 third argument.
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_generic_3( unsigned int i_instr,
                                                  unsigned int i_arg0,
                                                  unsigned int i_arg1,
                                                  unsigned int i_arg2 );

/**
 * Generates a generic POWER-instruction with four arguments
 * following the syntax of the mnemonics.
 *
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_arg0 first argument.
 * @param i_arg1 second argument.
 * @param i_arg2 third argument.
 * @param i_arg3 fourth argument
 * @return machine code.
 **/
LIBXSMM_API_INTERN
unsigned int libxsmm_power_instruction_generic_4( unsigned int i_instr,
                                                  unsigned int i_arg0,
                                                  unsigned int i_arg1,
                                                  unsigned int i_arg2,
                                                  unsigned int i_arg3 );

/**
 * Generates a generic POWER-instruction with two arguments
 * following the syntax of the mnemonics.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_arg0 first argument.
 * @param i_arg1 second argument.
 **/
LIBXSMM_API_INTERN
void libxsmm_power_instruction_2( libxsmm_generated_code * io_generated_code,
                                  unsigned int             i_instr,
                                  unsigned int             i_arg0,
                                  unsigned int             i_arg1 );

/**
 * Generates a generic POWER-instruction with three arguments
 * following the syntax of the mnemonics.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_arg0 first argument.
 * @param i_arg1 second argument.
 * @param i_arg2 third argument.
 **/
LIBXSMM_API_INTERN
void libxsmm_power_instruction_3( libxsmm_generated_code * io_generated_code,
                                  unsigned int             i_instr,
                                  unsigned int             i_arg0,
                                  unsigned int             i_arg1,
                                  unsigned int             i_arg2 );

/**
 * Generates a generic POWER-instruction with four arguments
 * following the syntax of the mnemonics.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_instr input-instruction with zeroed arguments.
 * @param i_arg0 first argument.
 * @param i_arg1 second argument.
 * @param i_arg2 third argument.
 * @param i_arg3 fourth argument.
 **/
LIBXSMM_API_INTERN
void libxsmm_power_instruction_4( libxsmm_generated_code * io_generated_code,
                                  unsigned int             i_instr,
                                  unsigned int             i_arg0,
                                  unsigned int             i_arg1,
                                  unsigned int             i_arg2,
                                  unsigned int             i_arg3 );

/**
 * Opens the inline assembly section / jit stream.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_gprMax max general purpose register which is saved on the stack.
 * @param i_fprMax max floating point register which is saved on the stack.
 * @param i_vsrMax max vector register which is saved on the stack.
 **/
LIBXSMM_API_INTERN
void libxsmm_power_instruction_open_stream( libxsmm_generated_code * io_generated_code,
                                            unsigned short           i_gprMax,
                                            unsigned short           i_fprMax,
                                            unsigned short           i_vsrMax );

/**
 * Closes the inline assembly section / jit stream.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param i_gprMax max general purpose register which is restored from the stack.
 * @param i_fprMax max floating point register which is restored from the stack.
 * @param i_vsrMax max vector register which is restored from the stack.
 **/
LIBXSMM_API_INTERN
void libxsmm_power_instruction_close_stream( libxsmm_generated_code * io_generated_code,
                                             unsigned short           i_gprMax,
                                             unsigned short           i_fprMax,
                                             unsigned short           i_vsrMax );

/**
 * Generates a label to which one can jump back and pushes it on the loop label stack.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure.
 * @param io_loop_label_tracker data structure to handle loop labels, nested loops are supported, but not overlapping loops.
*/
LIBXSMM_API_INTERN
void libxsmm_power_instruction_register_jump_back_label( libxsmm_generated_code     * io_generated_code,
                                                         libxsmm_loop_label_tracker * io_loop_label_tracker );

/**
 * Pops the latest from the loop label stack and jumps there based on the condition.
 *
 * @param io_generated_code pointer to the pointer of the generated code structure
 * @param i_gprs  GPR which is compared to zero.
 * @param io_loop_label_tracker data structure to handle loop labels will jump to latest registered label.
*/
LIBXSMM_API_INTERN
void libxsmm_power_instruction_cond_jump_back_to_label( libxsmm_generated_code     * io_generated_code,
                                                        unsigned int                 i_gpr,
                                                        libxsmm_loop_label_tracker * io_loop_label_tracker );


#endif