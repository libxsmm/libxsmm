/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#include "generator_common.h"
#include "libxsmm_main.h"

#if !defined(GENERATOR_COMMON_MAX_ERROR_LENGTH)
# define GENERATOR_COMMON_MAX_ERROR_LENGTH 511
#endif


LIBXSMM_API_INLINE
void libxsmm_strncpy( char*                  o_dest,
                      const char*            i_src,
                      unsigned int           i_dest_length,
                      unsigned int           i_src_length ) {
  if ( i_dest_length < i_src_length ) {
    fprintf( stderr, "LIBXSMM fatal error: libxsmm_strncpy destination buffer is too small!\n" );
    exit(-1);
  }

  /* @TODO check for aliasing? */

  strcpy( o_dest, i_src );
}

LIBXSMM_API_INTERN
void libxsmm_append_code_as_string( libxsmm_generated_code* io_generated_code,
                                    const char*             i_code_to_append,
                                    const int               i_append_length ) {
  size_t l_length_1 = 0;
  size_t l_length_2 = 0;
  char* l_new_string = NULL;
  char* current_code = (char*)io_generated_code->generated_code;

  /* check if end up here accidentally */
  if ( io_generated_code->code_type > 1 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_APPEND_STR );
    return;
  }

  /* some safety checks */
  if (current_code != NULL) {
    l_length_1 = io_generated_code->code_size;
  } else {
    /* nothing to do */
    l_length_1 = 0;
  }
  if (i_code_to_append != NULL) {
    l_length_2 = i_append_length;
  } else {
    fprintf(stderr, "LIBXSMM WARNING libxsmm_append_code_as_string was called with an empty string for appending code" );
  }

  /* allocate new string */
  l_new_string = (char*) malloc( (l_length_1+l_length_2+1)*sizeof(char) );
  if (l_new_string == NULL) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ALLOC );
    return;
  }

  /* copy old content */
  if (l_length_1 > 0) {
    /* @TODO using memcpy instead? */
    libxsmm_strncpy( l_new_string, current_code, (unsigned int)(l_length_1+l_length_2), (unsigned int)l_length_1 );
  } else {
    l_new_string[0] = '\0';
  }

  /* append new string */
  /* @TODO using memcpy instead? */
  if (i_code_to_append != NULL) {
    strcat(l_new_string, i_code_to_append);
  }

  /* free old memory and overwrite pointer */
  if (l_length_1 > 0)
    free(current_code);

  io_generated_code->generated_code = (void*)l_new_string;

  /* update counters */
  io_generated_code->code_size = (unsigned int)(l_length_1+l_length_2);
  io_generated_code->buffer_size = (io_generated_code->code_size) + 1;
}

LIBXSMM_API_INTERN
void libxsmm_close_function( libxsmm_generated_code* io_generated_code ) {
  if ( io_generated_code->code_type == 0 ) {
    char l_new_code[512];
    const int l_max_code_length = 511;
    const int l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "}\n\n" );
    libxsmm_append_code_as_string(io_generated_code, l_new_code, l_code_length );
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_check_x86_gp_reg_name_callee_save( const unsigned int i_gp_reg_number ) {
  if ( (i_gp_reg_number == LIBXSMM_X86_GP_REG_RBX) ||
       (i_gp_reg_number == LIBXSMM_X86_GP_REG_RBP) ||
       (i_gp_reg_number == LIBXSMM_X86_GP_REG_R12) ||
       (i_gp_reg_number == LIBXSMM_X86_GP_REG_R13) ||
       (i_gp_reg_number == LIBXSMM_X86_GP_REG_R14) ||
       (i_gp_reg_number == LIBXSMM_X86_GP_REG_R15) ) {
    return 1;
  } else {
    return 0;
  }
}

LIBXSMM_API_INTERN
void libxsmm_get_x86_gp_reg_name( const unsigned int i_gp_reg_number,
                                  char*              o_gp_reg_name,
                                  const int          i_gp_reg_name_max_length ) {
  switch (i_gp_reg_number) {
    case LIBXSMM_X86_GP_REG_RAX:
      libxsmm_strncpy(o_gp_reg_name, "rax", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_RCX:
      libxsmm_strncpy(o_gp_reg_name, "rcx", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_RDX:
      libxsmm_strncpy(o_gp_reg_name, "rdx", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_RBX:
      libxsmm_strncpy(o_gp_reg_name, "rbx", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_RSP:
      libxsmm_strncpy(o_gp_reg_name, "rsp", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_RBP:
      libxsmm_strncpy(o_gp_reg_name, "rbp", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_RSI:
      libxsmm_strncpy(o_gp_reg_name, "rsi", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_RDI:
      libxsmm_strncpy(o_gp_reg_name, "rdi", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_R8:
      libxsmm_strncpy(o_gp_reg_name, "r8", i_gp_reg_name_max_length, 2 );
      break;
    case LIBXSMM_X86_GP_REG_R9:
      libxsmm_strncpy(o_gp_reg_name, "r9", i_gp_reg_name_max_length, 2 );
      break;
    case LIBXSMM_X86_GP_REG_R10:
      libxsmm_strncpy(o_gp_reg_name, "r10", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_R11:
      libxsmm_strncpy(o_gp_reg_name, "r11", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_R12:
      libxsmm_strncpy(o_gp_reg_name, "r12", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_R13:
      libxsmm_strncpy(o_gp_reg_name, "r13", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_R14:
      libxsmm_strncpy(o_gp_reg_name, "r14", i_gp_reg_name_max_length, 3 );
      break;
    case LIBXSMM_X86_GP_REG_R15:
      libxsmm_strncpy(o_gp_reg_name, "r15", i_gp_reg_name_max_length, 3 );
      break;
    default:
      fprintf(stderr, "libxsmm_get_x86_64_gp_req_name i_gp_reg_number is out of range!\n");
      exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_get_x86_instr_name( const unsigned int i_instr_number,
                                 char*              o_instr_name,
                                 const int          i_instr_name_max_length ) {
  switch (i_instr_number) {
    /* AVX vector moves */
    case LIBXSMM_X86_INSTR_VMOVAPD:
      libxsmm_strncpy(o_instr_name, "vmovapd", i_instr_name_max_length, 7 );
      break;
    case LIBXSMM_X86_INSTR_VMOVUPD:
      libxsmm_strncpy(o_instr_name, "vmovupd", i_instr_name_max_length, 7 );
      break;
    case LIBXSMM_X86_INSTR_VMOVAPS:
      libxsmm_strncpy(o_instr_name, "vmovaps", i_instr_name_max_length, 7 );
      break;
    case LIBXSMM_X86_INSTR_VMOVUPS:
      libxsmm_strncpy(o_instr_name, "vmovups", i_instr_name_max_length, 7 );
      break;
    case LIBXSMM_X86_INSTR_VBROADCASTSD:
      libxsmm_strncpy(o_instr_name, "vbroadcastsd", i_instr_name_max_length, 12 );
      break;
    case LIBXSMM_X86_INSTR_VBROADCASTSS:
      libxsmm_strncpy(o_instr_name, "vbroadcastss", i_instr_name_max_length, 12 );
      break;
    case LIBXSMM_X86_INSTR_VMOVDDUP:
      libxsmm_strncpy(o_instr_name, "vmovddup", i_instr_name_max_length, 8 );
      break;
    case LIBXSMM_X86_INSTR_VMOVSD:
      libxsmm_strncpy(o_instr_name, "vmovsd", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VMOVSS:
      libxsmm_strncpy(o_instr_name, "vmovss", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VPBROADCASTB:
      libxsmm_strncpy(o_instr_name, "vpbroadcastb", i_instr_name_max_length, 12 );
      break;
    case LIBXSMM_X86_INSTR_VPBROADCASTW:
      libxsmm_strncpy(o_instr_name, "vpbroadcastw", i_instr_name_max_length, 12 );
      break;
    case LIBXSMM_X86_INSTR_VPBROADCASTD:
      libxsmm_strncpy(o_instr_name, "vpbroadcastd", i_instr_name_max_length, 12 );
      break;
    case LIBXSMM_X86_INSTR_VPBROADCASTQ:
      libxsmm_strncpy(o_instr_name, "vpbroadcastq", i_instr_name_max_length, 12 );
      break;
    /* SSE vector moves */
    case LIBXSMM_X86_INSTR_MOVAPD:
      libxsmm_strncpy(o_instr_name, "movapd", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_MOVUPD:
      libxsmm_strncpy(o_instr_name, "movupd", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_MOVAPS:
      libxsmm_strncpy(o_instr_name, "movaps", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_MOVUPS:
      libxsmm_strncpy(o_instr_name, "movups", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_MOVDDUP:
      libxsmm_strncpy(o_instr_name, "movddup", i_instr_name_max_length, 7 );
      break;
    case LIBXSMM_X86_INSTR_MOVSD:
      libxsmm_strncpy(o_instr_name, "movsd", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_MOVSS:
      libxsmm_strncpy(o_instr_name, "movss", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_SHUFPS:
      libxsmm_strncpy(o_instr_name, "shufps", i_instr_name_max_length, 6 );
      break;
    /* Gather/scatter single precision */
    case LIBXSMM_X86_INSTR_VGATHERDPS:
      libxsmm_strncpy(o_instr_name, "vgatherdps", i_instr_name_max_length, 10 );
      break;
    case LIBXSMM_X86_INSTR_VGATHERQPS:
      libxsmm_strncpy(o_instr_name, "vgatherqps", i_instr_name_max_length, 10 );
      break;
    case LIBXSMM_X86_INSTR_VSCATTERDPS:
      libxsmm_strncpy(o_instr_name, "vscatterdps", i_instr_name_max_length, 11 );
      break;
    case LIBXSMM_X86_INSTR_VSCATTERQPS:
      libxsmm_strncpy(o_instr_name, "vscatterqps", i_instr_name_max_length, 11 );
      break;
    /* Gather/scatter double precision */
    case LIBXSMM_X86_INSTR_VGATHERDPD:
      libxsmm_strncpy(o_instr_name, "vgatherdpd", i_instr_name_max_length, 10 );
      break;
    case LIBXSMM_X86_INSTR_VGATHERQPD:
      libxsmm_strncpy(o_instr_name, "vgatherqpd", i_instr_name_max_length, 10 );
      break;
    case LIBXSMM_X86_INSTR_VSCATTERDPD:
      libxsmm_strncpy(o_instr_name, "vscatterdpd", i_instr_name_max_length, 11 );
      break;
    case LIBXSMM_X86_INSTR_VSCATTERQPD:
      libxsmm_strncpy(o_instr_name, "vscatterqpd", i_instr_name_max_length, 11 );
      break;
    /* AVX double precision */
    case LIBXSMM_X86_INSTR_VXORPD:
      libxsmm_strncpy(o_instr_name, "vxorpd", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VMULPD:
      libxsmm_strncpy(o_instr_name, "vmulpd", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VADDPD:
      libxsmm_strncpy(o_instr_name, "vaddpd", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VFMADD231PD:
      libxsmm_strncpy(o_instr_name, "vfmadd231pd", i_instr_name_max_length, 11 );
      break;
    case LIBXSMM_X86_INSTR_VMULSD:
      libxsmm_strncpy(o_instr_name, "vmulsd", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VADDSD:
      libxsmm_strncpy(o_instr_name, "vaddsd", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VFMADD231SD:
      libxsmm_strncpy(o_instr_name, "vfmadd231sd", i_instr_name_max_length, 11 );
      break;
    /* AVX single precision */
    case LIBXSMM_X86_INSTR_VXORPS:
      libxsmm_strncpy(o_instr_name, "vxorps", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VMULPS:
      libxsmm_strncpy(o_instr_name, "vmulps", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VADDPS:
      libxsmm_strncpy(o_instr_name, "vaddps", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VFMADD231PS:
      libxsmm_strncpy(o_instr_name, "vfmadd231ps", i_instr_name_max_length, 11 );
      break;
    case LIBXSMM_X86_INSTR_VMULSS:
      libxsmm_strncpy(o_instr_name, "vmulss", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VADDSS:
      libxsmm_strncpy(o_instr_name, "vaddss", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VFMADD231SS:
      libxsmm_strncpy(o_instr_name, "vfmadd231ss", i_instr_name_max_length, 11 );
      break;
    /* SSE double precision */
    case LIBXSMM_X86_INSTR_XORPD:
      libxsmm_strncpy(o_instr_name, "xorpd", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_MULPD:
      libxsmm_strncpy(o_instr_name, "mulpd", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_ADDPD:
      libxsmm_strncpy(o_instr_name, "addpd", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_MULSD:
      libxsmm_strncpy(o_instr_name, "mulsd", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_ADDSD:
      libxsmm_strncpy(o_instr_name, "addsd", i_instr_name_max_length, 5 );
      break;
    /* SSE single precision */
    case LIBXSMM_X86_INSTR_XORPS:
      libxsmm_strncpy(o_instr_name, "xorps", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_MULPS:
      libxsmm_strncpy(o_instr_name, "mulps", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_ADDPS:
      libxsmm_strncpy(o_instr_name, "addps", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_MULSS:
      libxsmm_strncpy(o_instr_name, "mulss", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_ADDSS:
      libxsmm_strncpy(o_instr_name, "addss", i_instr_name_max_length, 5 );
      break;
    /* XOR AVX512F */
    case LIBXSMM_X86_INSTR_VPXORD:
      libxsmm_strncpy(o_instr_name, "vpxord", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VPADDB:
      libxsmm_strncpy(o_instr_name, "vpaddb", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VPADDW:
      libxsmm_strncpy(o_instr_name, "vpaddw", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VPADDD:
      libxsmm_strncpy(o_instr_name, "vpaddd", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VPADDQ:
      libxsmm_strncpy(o_instr_name, "vpaddq", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VPMADDWD:
      libxsmm_strncpy(o_instr_name, "vpmaddwd", i_instr_name_max_length, 8 );
      break;
    case LIBXSMM_X86_INSTR_VPMADDUBSW:
      libxsmm_strncpy(o_instr_name, "vpmaddubsw", i_instr_name_max_length, 10 );
      break;
    case LIBXSMM_X86_INSTR_VPSRAVD:
      libxsmm_strncpy(o_instr_name, "vpsravd", i_instr_name_max_length, 7 );
      break;
    case LIBXSMM_X86_INSTR_VPSRAD:
      libxsmm_strncpy(o_instr_name, "vpsrad", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VPSLLD:
      libxsmm_strncpy(o_instr_name, "vpslld", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VPCMPD:
      libxsmm_strncpy(o_instr_name, "vpcmpd", i_instr_name_max_length, 6 );
      break;
    /* AVX512, QFMA */
    case LIBXSMM_X86_INSTR_V4FMADDPS:
      libxsmm_strncpy(o_instr_name, "v4fmaddps", i_instr_name_max_length, 9 );
      break;
    case LIBXSMM_X86_INSTR_V4FNMADDPS:
      libxsmm_strncpy(o_instr_name, "v4fnmaddps", i_instr_name_max_length, 10 );
      break;
    case LIBXSMM_X86_INSTR_V4FMADDSS:
      libxsmm_strncpy(o_instr_name, "v4fmaddss", i_instr_name_max_length, 9 );
      break;
    case LIBXSMM_X86_INSTR_V4FNMADDSS:
      libxsmm_strncpy(o_instr_name, "v4fnmaddss", i_instr_name_max_length, 10 );
      break;
    case LIBXSMM_X86_INSTR_VP4DPWSSD:
      libxsmm_strncpy(o_instr_name, "vp4dpwssd", i_instr_name_max_length, 9 );
      break;
    case LIBXSMM_X86_INSTR_VP4DPWSSDS:
      libxsmm_strncpy(o_instr_name, "vp4dpwssds", i_instr_name_max_length, 10 );
      break;
    /* AVX512, VNNI */
    case LIBXSMM_X86_INSTR_VPDPWSSD:
      libxsmm_strncpy(o_instr_name, "vpdpwssd", i_instr_name_max_length, 8 );
      break;
    case LIBXSMM_X86_INSTR_VPDPWSSDS:
      libxsmm_strncpy(o_instr_name, "vpdpwssds", i_instr_name_max_length, 9 );
      break;
    case LIBXSMM_X86_INSTR_VPDPBUSD:
      libxsmm_strncpy(o_instr_name, "vpdpbusd", i_instr_name_max_length, 8 );
      break;
    case LIBXSMM_X86_INSTR_VPDPBUSDS:
      libxsmm_strncpy(o_instr_name, "vpdpbusds", i_instr_name_max_length, 9 );
      break;
    /* AVX512, BF16 */
    case LIBXSMM_X86_INSTR_VDPBF16PS:
      libxsmm_strncpy(o_instr_name, "vdpbf16ps", i_instr_name_max_length, 9 );
      break;
    case LIBXSMM_X86_INSTR_VCVTNEPS2BF16:
      libxsmm_strncpy(o_instr_name, "vcvtneps2bf16", i_instr_name_max_length, 13 );
      break;
    case LIBXSMM_X86_INSTR_VCVTNE2PS2BF16:
      libxsmm_strncpy(o_instr_name, "vcvtne2ps2bf16", i_instr_name_max_length, 14 );
      break;
    /* GP instructions */
    case LIBXSMM_X86_INSTR_ADDQ:
      libxsmm_strncpy(o_instr_name, "addq", i_instr_name_max_length, 4 );
      break;
    case LIBXSMM_X86_INSTR_SUBQ:
      libxsmm_strncpy(o_instr_name, "subq", i_instr_name_max_length, 4 );
      break;
    case LIBXSMM_X86_INSTR_MOVQ:
      libxsmm_strncpy(o_instr_name, "movq", i_instr_name_max_length, 4 );
      break;
    case LIBXSMM_X86_INSTR_CMPQ:
      libxsmm_strncpy(o_instr_name, "cmpq", i_instr_name_max_length, 4 );
      break;
    case LIBXSMM_X86_INSTR_JL:
      libxsmm_strncpy(o_instr_name, "jl", i_instr_name_max_length, 2 );
      break;
    case LIBXSMM_X86_INSTR_JE:
      libxsmm_strncpy(o_instr_name, "je", i_instr_name_max_length, 2 );
      break;
    case LIBXSMM_X86_INSTR_JZ:
      libxsmm_strncpy(o_instr_name, "jz", i_instr_name_max_length, 2 );
      break;
    case LIBXSMM_X86_INSTR_JG:
      libxsmm_strncpy(o_instr_name, "jg", i_instr_name_max_length, 2 );
      break;
    case LIBXSMM_X86_INSTR_JNE:
      libxsmm_strncpy(o_instr_name, "jne", i_instr_name_max_length, 3 );
      break;
    case LIBXSMM_X86_INSTR_JNZ:
      libxsmm_strncpy(o_instr_name, "jnz", i_instr_name_max_length, 3 );
      break;
    case LIBXSMM_X86_INSTR_JGE:
      libxsmm_strncpy(o_instr_name, "jge", i_instr_name_max_length, 3 );
      break;
    case LIBXSMM_X86_INSTR_JLE:
      libxsmm_strncpy(o_instr_name, "jle", i_instr_name_max_length, 3 );
      break;
    case LIBXSMM_X86_INSTR_PREFETCHT0:
      libxsmm_strncpy(o_instr_name, "prefetcht0", i_instr_name_max_length, 10 );
      break;
    case LIBXSMM_X86_INSTR_PREFETCHT1:
      libxsmm_strncpy(o_instr_name, "prefetcht1", i_instr_name_max_length, 10 );
      break;
    case LIBXSMM_X86_INSTR_PREFETCHT2:
      libxsmm_strncpy(o_instr_name, "prefetcht2", i_instr_name_max_length, 10 );
      break;
    case LIBXSMM_X86_INSTR_PREFETCHNTA:
      libxsmm_strncpy(o_instr_name, "prefetchnta", i_instr_name_max_length, 11 );
      break;
    case LIBXSMM_X86_INSTR_KMOV:
      libxsmm_strncpy(o_instr_name, "kmov", i_instr_name_max_length, 4 );
      break;
    case LIBXSMM_X86_INSTR_KMOVW:
      libxsmm_strncpy(o_instr_name, "kmovw", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_KMOVB:
      libxsmm_strncpy(o_instr_name, "kmovb", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_KMOVD:
      libxsmm_strncpy(o_instr_name, "kmovd", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_KMOVQ:
      libxsmm_strncpy(o_instr_name, "kmovq", i_instr_name_max_length, 5 );
      break;
    case LIBXSMM_X86_INSTR_KXNORW:
      libxsmm_strncpy(o_instr_name, "kxnorw", i_instr_name_max_length, 6 );
      break;
    case LIBXSMM_X86_INSTR_VMOVNTPD:
      libxsmm_strncpy(o_instr_name, "vmovntpd", i_instr_name_max_length, 8 );
      break;
    case LIBXSMM_X86_INSTR_VMOVNTPS:
      libxsmm_strncpy(o_instr_name, "vmovntps", i_instr_name_max_length, 8 );
      break;
    case LIBXSMM_X86_INSTR_VMOVNTDQ:
      libxsmm_strncpy(o_instr_name, "vmovntdq", i_instr_name_max_length, 8 );
      break;
    /* default, we didn't had a match */
    default:
      fprintf(stderr, "libxsmm_get_x86_64_instr_name i_instr_number (%u) is out of range!\n", i_instr_number);
      exit(-1);
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_is_x86_vec_instr_single_precision( const unsigned int i_instr_number ) {
  unsigned int l_return = 0;

  switch (i_instr_number) {
    case LIBXSMM_X86_INSTR_VMOVAPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMOVUPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMOVAPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VMOVUPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VBROADCASTSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VBROADCASTSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VMOVDDUP:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMOVSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMOVSS:
      l_return = 1;
      break;
    /* SSE vector moves */
    case LIBXSMM_X86_INSTR_MOVAPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MOVUPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MOVAPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_MOVUPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_MOVDDUP:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MOVSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MOVSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_SHUFPS:
      l_return = 1;
      break;
    /* Gather/Scatter single precision */
    case LIBXSMM_X86_INSTR_VGATHERDPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VGATHERQPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VSCATTERDPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VSCATTERQPS:
      l_return = 1;
      break;
    /* Gather/Scatter double precision */
    case LIBXSMM_X86_INSTR_VGATHERDPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VGATHERQPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VSCATTERDPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VSCATTERQPD:
      l_return = 0;
      break;
    /* AVX double precision */
    case LIBXSMM_X86_INSTR_VXORPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMULPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VADDPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VFMADD231PD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VMULSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VADDSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_VFMADD231SD:
      l_return = 0;
      break;
    /* AVX single precision */
    case LIBXSMM_X86_INSTR_VXORPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VMULPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VADDPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VFMADD231PS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VMULSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VADDSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VFMADD231SS:
      l_return = 1;
      break;
    /* SSE double precision */
    case LIBXSMM_X86_INSTR_XORPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MULPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_ADDPD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_MULSD:
      l_return = 0;
      break;
    case LIBXSMM_X86_INSTR_ADDSD:
      l_return = 0;
      break;
    /* SSE single precision */
    case LIBXSMM_X86_INSTR_XORPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_MULPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_ADDPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_MULSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_ADDSS:
      l_return = 1;
      break;
    /* AVX512, QFMA */
    case LIBXSMM_X86_INSTR_V4FMADDPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_V4FNMADDPS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_V4FMADDSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_V4FNMADDSS:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VP4DPWSSD:
      l_return = 1;
      break;
    case LIBXSMM_X86_INSTR_VP4DPWSSDS:
      l_return = 1;
      break;
    /* default, we didn't had a match */
    default:
      fprintf(stderr, "libxsmm_is_x86_vec_instr_single_precision i_instr_number (%u) is not a x86 FP vector instruction!\n", i_instr_number);
      exit(-1);
  }

  return l_return;
}

LIBXSMM_API_INTERN
void libxsmm_reset_x86_gp_reg_mapping( libxsmm_gp_reg_mapping* io_gp_reg_mapping ) {
  io_gp_reg_mapping->gp_reg_a = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_b = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_c = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
/*  io_gp_reg_mapping->gp_reg_c_prefetch = LIBXSMM_X86_GP_REG_UNDEF;*/
  io_gp_reg_mapping->gp_reg_mloop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_nloop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_kloop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_reduce_count = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_reduce_loop = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_scf = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  io_gp_reg_mapping->gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
}

LIBXSMM_API_INTERN
void libxsmm_reset_loop_label_tracker( libxsmm_loop_label_tracker* io_loop_label_tracker ) {
  memset(io_loop_label_tracker, 0, sizeof(*io_loop_label_tracker));
}

LIBXSMM_API_INTERN
void libxsmm_reset_jump_label_tracker( libxsmm_jump_label_tracker* io_jump_label_tracker ) {
  memset(io_jump_label_tracker, 0, sizeof(*io_jump_label_tracker));
}

LIBXSMM_API_INTERN
void libxsmm_mmfunction_signature( libxsmm_generated_code*         io_generated_code,
                                   const char*                     i_routine_name,
                                   const libxsmm_gemm_descriptor*  i_xgemm_desc ) {
  char l_new_code[512];
  int l_max_code_length = 511;
  int l_code_length = 0;

  LIBXSMM_ASSERT_MSG(NULL != i_xgemm_desc, "Invalid descriptor");
  if ( io_generated_code->code_type > 1 ) {
    return;
  } else if ( io_generated_code->code_type == 1 ) {
    l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, ".global %s\n.type %s, @function\n%s:\n", i_routine_name, i_routine_name, i_routine_name);
  } else {
    /* selecting the correct signature */
    if (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      if (LIBXSMM_GEMM_PREFETCH_NONE == i_xgemm_desc->prefetch) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const float* A, const float* B, float* C) {\n", i_routine_name);
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const float* A, const float* B, float* C, const float* A_prefetch, const float* B_prefetch, const float* C_prefetch) {\n", i_routine_name);
      }
    } else if (LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      if (LIBXSMM_GEMM_PREFETCH_NONE == i_xgemm_desc->prefetch) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const double* A, const double* B, double* C) {\n", i_routine_name);
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const double* A, const double* B, double* C, const double* A_prefetch, const double* B_prefetch, const double* C_prefetch) {\n", i_routine_name);
      }
    } else if (LIBXSMM_GEMM_PRECISION_I16 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) ) {
      if (LIBXSMM_GEMM_PREFETCH_NONE == i_xgemm_desc->prefetch) {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const short* A, const short* B, int* C) {\n", i_routine_name);
      } else {
        l_code_length = LIBXSMM_SNPRINTF(l_new_code, l_max_code_length, "void %s(const short* A, const short* B, int* C, const short* A_prefetch, const short* B_prefetch, const int* C_prefetch) {\n", i_routine_name);
      }
    } else {}
  }

  libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
}

LIBXSMM_API_INTERN
void libxsmm_generator_isa_check_header( libxsmm_generated_code* io_generated_code ) {
  if ( io_generated_code->code_type == 0 ) {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    if ( io_generated_code->arch <= LIBXSMM_X86_SSE4 ) {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifdef __SSE3__\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifdef __AVX__\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling SSE3 code on AVX or newer architecture: \" __FILE__)\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#endif\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else if ( io_generated_code->arch == LIBXSMM_X86_AVX ) {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifdef __AVX__\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifdef __AVX2__\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling AVX code on AVX2 or newer architecture: \" __FILE__)\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#endif\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else if ( io_generated_code->arch == LIBXSMM_X86_AVX2 ) {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifdef __AVX2__\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifdef __AVX512F__\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling AVX2 code on AVX512 or newer architecture: \" __FILE__)\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#endif\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#ifdef __AVX512F__\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else if ( io_generated_code->arch < LIBXSMM_X86_SSE3 ) {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#pragma message (\"LIBXSMM KERNEL COMPILATION WARNING: compiling arch-independent gemm kernel in: \" __FILE__)\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_isa_check_footer( libxsmm_generated_code* io_generated_code ) {
  if ( io_generated_code->code_type == 0 ) {
    char l_new_code[512];
    int l_max_code_length = 511;
    int l_code_length = 0;

    if ( ( io_generated_code->arch >= LIBXSMM_X86_SSE3 ) && ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT )  )
    {
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#else\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#pragma message (\"LIBXSMM KERNEL COMPILATION ERROR in: \" __FILE__)\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#error No kernel was compiled, lacking support for current architecture?\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
      l_code_length = LIBXSMM_SNPRINTF( l_new_code, l_max_code_length, "#endif\n\n" );
      libxsmm_append_code_as_string( io_generated_code, l_new_code, l_code_length );
    } else if ( io_generated_code->arch < LIBXSMM_X86_SSE3 ) {
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
      return;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_handle_error( libxsmm_generated_code* io_generated_code,
                           const unsigned int      i_error_code,
                           const char* context,
                           int emit_message ) {
  static LIBXSMM_TLS unsigned int last_error_code;
  if (i_error_code != last_error_code) {
    if (0 != emit_message) {
      LIBXSMM_STDIO_ACQUIRE();
      if (0 != context && 0 != *context && '0' != *context) {
        fprintf(stderr, "LIBXSMM ERROR (%s): %s\n", context, libxsmm_strerror(i_error_code));
      }
      else {
        fprintf(stderr, "LIBXSMM ERROR: %s\n", libxsmm_strerror(i_error_code));
      }
      LIBXSMM_STDIO_RELEASE();
    }
    last_error_code = i_error_code;
  }
  io_generated_code->last_error = i_error_code;
}

LIBXSMM_API
const char* libxsmm_strerror(unsigned int i_error_code) {
  static LIBXSMM_TLS char error_message[GENERATOR_COMMON_MAX_ERROR_LENGTH+1];

  switch (i_error_code) {
    case LIBXSMM_ERR_GENERAL:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "a general error occurred (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_ALLOC:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "memory allocation failed (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_BUFFER_TOO_SMALL:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "code generation ran out of buffer capacity (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_APPEND_STR:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "append code as string was called for generation mode which does not support this (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_ARCH_PREC:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "unknown architecture or unsupported precision (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_ARCH:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "unknown architecture (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_UNSUP_ARCH:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "unsupported arch for the selected module was specified (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_LDA:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "lda needs to be greater than or equal to m (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_LDB:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "ldb needs to be greater than or equal to k (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_LDC:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "ldc needs to be greater than or equal to m (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_SPGEMM_GEN:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "could not determine which sparse code generation variant is requested (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSC_INPUT:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "could not open the CSC input file, or invalid file content found (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSC_READ_LEN:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "exceeded predefined line-length when reading line of CSC file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSC_READ_DESC:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "error when reading descriptor of CSC file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSC_READ_ELEMS:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "error when reading line of CSC file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSC_LEN:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "number of elements read differs from number of elements specified in CSC file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_N_BLOCK:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "invalid N blocking in microkernel (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_M_BLOCK:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "invalid M blocking in microkernel (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_K_BLOCK:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "invalid K blocking in microkernel (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_REG_BLOCK:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "invalid MxN register blocking was specified (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_NO_AVX512_BCAST:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "fused memory broadcast is not supported on other platforms than AVX512 (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_NO_AVX512_QFMA:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "there is no QFMA instruction set extension available (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CALLEE_SAVE_A:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "reg_a cannot be callee save, since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CALLEE_SAVE_B:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "reg_b cannot be callee save, since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CALLEE_SAVE_C:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "reg_c cannot be callee save, since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CALLEE_SAVE_A_PREF:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "reg_a_prefetch cannot be callee save, since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CALLEE_SAVE_B_PREF:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "reg_b_prefetch cannot be callee save, since input, please use either rdi, rsi, rdx, rcx, r8, r9 for this value (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_NO_INDEX_SCALE_ADDR:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "Index + Scale addressing mode is currently not implemented (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_UNSUPPORTED_JUMP:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "Unsupported jump instruction requested (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_NO_JMPLBL_AVAIL:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "No destination jump label is available (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_EXCEED_JMPLBL:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "too many nested loops, exceeding loop label tracker (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_JMPLBL_USED:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "attempted to use an already used jump label (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSC_ALLOC_DATA:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "could not allocate temporary memory for reading CSC file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSR_ALLOC_DATA:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "could not allocate temporary memory for reading CSR file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSR_INPUT:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "could not open the specified CSR input file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSR_READ_LEN:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "exceeded predefined line-length when reading line of CSR file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSR_READ_DESC:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "error when reading descriptor of CSR file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSR_READ_ELEMS:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "error when reading line of CSR file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_CSR_LEN:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "number of elements read differs from number of elements specified in CSR file (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_UNSUP_DATATYPE:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "unsupported datatype was requested (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_UNSUP_DT_FORMAT:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "unsupported datatype and format combination was requested (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_INVALID_GEMM_CONFIG:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "invalid GEMM config in setup detected (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_UNIQUE_VAL:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "for sparse-A in reg: too many values in A (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_VEC_REG_MUST_BE_UNDEF:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "input vector register parameter must be undefined here (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_TRANS_B:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "GEMM kernel with trans B requested, but target/datatype not supported (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_LDB_TRANS:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "ldb needs to be greater than or equal to n (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_VNNI_A:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "A is not provided in supported VNNI format (error #%u)!", i_error_code );
      break;
    case LIBXSMM_ERR_VNNI_B:
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "B is not provided in supported VNNI format (error #%u)!", i_error_code );
      break;
    default: /* we do not know what happened */
      LIBXSMM_SNPRINTF( error_message, GENERATOR_COMMON_MAX_ERROR_LENGTH,
        "an unknown error occurred (error #%u)!", i_error_code );
      break;
    }

  return error_message;
}


LIBXSMM_API_INTERN unsigned int libxsmm_compute_equalized_blocking(
  unsigned int i_size, unsigned int i_max_block,
  unsigned int* o_range_1, unsigned int* o_block_1,
  unsigned int* o_range_2, unsigned int* o_block_2 )
{
  unsigned int l_size = LIBXSMM_MAX(i_size, 1);
  unsigned int l_number_of_chunks = ((l_size - 1) / i_max_block) + 1;
  unsigned int l_modulo = l_size % l_number_of_chunks;
  unsigned int l_n2 = l_size / l_number_of_chunks;
  unsigned int l_n1 = l_n2 + 1;
  unsigned int l_N2 = 0;
  unsigned int l_N1 = 0;
  unsigned int l_chunk = 0;
  unsigned int l_ret = 0;

  /* ranges */
  if (l_n1 > i_max_block) l_n1 = i_max_block;
  for (l_chunk = 0; l_chunk < l_number_of_chunks; ++l_chunk) {
    if (l_chunk < l_modulo) {
      l_N1 += l_n1;
    } else {
      l_N2 += l_n2;
    }
  }

  /* if we have perfect blocking, swap n2 and n1 */
  if ( l_modulo == 0 ) {
    l_n1 = l_n2;
    l_N1 = l_N2;
    l_n2 = 0;
    l_N2 = 0;
  }

  /* some checks */
  if ( (l_N1 % l_n1) != 0 ) {
    l_ret = 1;
  }
  if ( l_n2 != 0 ) {
    if ( l_N2 % l_n2 != 0 ) {
      l_ret = 1;
    }
  }

  /* set output variables */
  *o_range_1 = l_N1;
  *o_block_1 = l_n1;
  *o_range_2 = l_N2;
  *o_block_2 = l_n2;

  return l_ret;
}

