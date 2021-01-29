/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <generator_x86_instructions.h>

/* #define XED_DECODE_TESTING  */

#ifdef XED_DECODE_TESTING
#include "xed/xed-interface.h"
#include "xed-examples-util.h"
#endif

#ifdef XED_DECODE_TESTING
void xed_initialization ( xed_decoded_inst_t *xedd_ptr )
{
  xed_state_t dstate;
  xed_uint_t bytes = 0;
  xed_decoded_inst_t xedd = *xedd_ptr;
  unsigned char itext[XED_MAX_INSTRUCTION_BYTES];
  xed_bool_t already_set_mode = 0;
  xed_chip_enum_t chip = XED_CHIP_INVALID;
  char const* decode_text=0;
  unsigned int len;
  xed_error_enum_t xed_error;
  unsigned int mpx_mode=0;
  unsigned int cet_mode=0;
  int iii, first;

  xed_tables_init();
  xed_set_verbosity( 99 );
  xed_state_zero(&dstate);
  dstate.mmode=XED_MACHINE_MODE_LONG_64;
  dstate.stack_addr_width=XED_ADDRESS_WIDTH_32b;
  xed_decoded_inst_zero_set_mode( xedd_ptr, &dstate);
  xed_decoded_inst_set_input_chip( xedd_ptr, chip);
  xed3_operand_set_mpxmode( xedd_ptr, mpx_mode);
  xed3_operand_set_cet( xedd_ptr, cet_mode);
}
#endif

#ifdef XED_DECODE_TESTING
void convert_xed_enums_to_libxsmm ( xed_reg_enum_t xreg, unsigned int *i_xsmm )
{
   if ( xreg == XED_REG_RAX ) *i_xsmm = LIBXSMM_X86_GP_REG_RAX;
   if ( xreg == XED_REG_RCX ) *i_xsmm = LIBXSMM_X86_GP_REG_RCX;
   if ( xreg == XED_REG_RDX ) *i_xsmm = LIBXSMM_X86_GP_REG_RDX;
   if ( xreg == XED_REG_RBX ) *i_xsmm = LIBXSMM_X86_GP_REG_RBX;
   if ( xreg == XED_REG_RSP ) *i_xsmm = LIBXSMM_X86_GP_REG_RSP;
   if ( xreg == XED_REG_RBP ) *i_xsmm = LIBXSMM_X86_GP_REG_RBP;
   if ( xreg == XED_REG_RSI ) *i_xsmm = LIBXSMM_X86_GP_REG_RSI;
   if ( xreg == XED_REG_RDI ) *i_xsmm = LIBXSMM_X86_GP_REG_RDI;
   if ( xreg == XED_REG_R8  ) *i_xsmm = LIBXSMM_X86_GP_REG_R8 ;
   if ( xreg == XED_REG_R9  ) *i_xsmm = LIBXSMM_X86_GP_REG_R9 ;
   if ( xreg == XED_REG_R10 ) *i_xsmm = LIBXSMM_X86_GP_REG_R10;
   if ( xreg == XED_REG_R11 ) *i_xsmm = LIBXSMM_X86_GP_REG_R11;
   if ( xreg == XED_REG_R12 ) *i_xsmm = LIBXSMM_X86_GP_REG_R12;
   if ( xreg == XED_REG_R13 ) *i_xsmm = LIBXSMM_X86_GP_REG_R13;
   if ( xreg == XED_REG_R14 ) *i_xsmm = LIBXSMM_X86_GP_REG_R14;
   if ( xreg == XED_REG_R15 ) *i_xsmm = LIBXSMM_X86_GP_REG_R15;
   if ( (xreg >= XED_REG_XMM0) && (xreg <= XED_REG_XMM31) ) *i_xsmm = xreg - XED_REG_XMM0;
   if ( (xreg >= XED_REG_YMM0) && (xreg <= XED_REG_YMM31) ) *i_xsmm = xreg - XED_REG_YMM0;
   if ( (xreg >= XED_REG_ZMM0) && (xreg <= XED_REG_ZMM31) ) *i_xsmm = xreg - XED_REG_ZMM0;

}
#endif

#ifdef XED_DECODE_TESTING
/* Returns -1 if xed can't decode */
int  xed_decode_to_libxsmm_parameters ( unsigned char *buf,
                                        unsigned int bytes,
                                        xed_decoded_inst_t *xedd_ptr,
                                        char *xed_instr_name,
                                        unsigned int *i_gp_reg_base,
                                        unsigned int *i_gp_reg_idx,
                                        unsigned int *i_scale,
                                        int *i_displacement,
                                        char *i_vector_name,
                                        unsigned int *i_vec_reg_number_0,
                                        unsigned int *i_vec_reg_number_1,
                                        unsigned int *i_vec_reg_number_2,
                                        unsigned int *i_mask_reg_number,
                                        unsigned int *i_use_zero_masking,
                                        unsigned int *i_is_store,
                                        unsigned short *i_imm8
                                      )
{
  xed_state_t dstate;
  xed_decoded_inst_t xedd = *xedd_ptr;
#define BUFLEN 1000
  char buffer[BUFLEN];
  unsigned char itext[XED_MAX_INSTRUCTION_BYTES];
  xed_bool_t already_set_mode = 0;
  xed_chip_enum_t chip = XED_CHIP_INVALID;
  char const* decode_text=0;
  unsigned int len;
  xed_error_enum_t xed_error;
  unsigned int mpx_mode=0;
  unsigned int cet_mode=0;
  xed_syntax_enum_t syntax;
  int i, ok, scale;
  unsigned int isyntax;
  xed_machine_mode_enum_t mmode;
  xed_address_width_enum_t stack_addr_width;
  xed_format_options_t format_options;
  xed_reg_enum_t base,indx;
  xed_decoded_inst_zero(&xedd);
  xed_decoded_inst_set_mode(&xedd, XED_MACHINE_MODE_LONG_64, XED_ADDRESS_WIDTH_64b);
  xed_decoded_inst_zero_set_mode(&xedd, &dstate);

  // One time initialization
  memset(&format_options,0, sizeof(format_options));
  format_options.hex_address_before_symbolic_name=0;
  format_options.xml_a=0;
  format_options.omit_unit_scale=0;
  format_options.no_sign_extend_signed_immediates=0;
  mmode=XED_MACHINE_MODE_LONG_64;
  stack_addr_width =XED_ADDRESS_WIDTH_64b;

  xed_format_set_options ( format_options );
  xed_decoded_inst_zero(&xedd);
  xed_decoded_inst_set_mode(&xedd, mmode, stack_addr_width);

  xed_error = xed_decode(&xedd,
                         XED_REINTERPRET_CAST(const xed_uint8_t*,buf),
                         bytes );

  if ( xed_error != XED_ERROR_NONE ) return (-1);

  strcpy ( xed_instr_name, xed_iclass_enum_t2str( xed_decoded_inst_get_iclass(&xedd)));
  for ( i = 0 ; i < strlen(xed_instr_name); i++ ) {
     if ( xed_instr_name[i] >= 'A' && xed_instr_name[i] <= 'Z' ) {
        xed_instr_name[i]= xed_instr_name[i] + 'a' - 'A';
     }
  }

  *i_gp_reg_base = LIBXSMM_X86_GP_REG_UNDEF;
  *i_gp_reg_idx  = LIBXSMM_X86_GP_REG_UNDEF;
  *i_scale = 0;
  *i_displacement = 0;
  *i_use_zero_masking = 0;
  *i_mask_reg_number = 0;
  *i_imm8 = LIBXSMM_X86_IMM_UNDEF;

  base = xed_decoded_inst_get_base_reg(&xedd,0);
  if ( base != XED_REG_INVALID ) {
     convert_xed_enums_to_libxsmm ( base, i_gp_reg_base );

     indx = xed_decoded_inst_get_index_reg(&xedd,0);
     if ( (indx != XED_REG_INVALID) && (indx != XED_REG_RSP) ) {
        convert_xed_enums_to_libxsmm ( indx, i_gp_reg_idx );
        *i_scale = xed_decoded_inst_get_scale( &xedd, 0 );
     }
  }

  /* Note: Displacements of 0 are NOT counted in XED as a displacement */
  if (xed_operand_values_has_memory_displacement(&xedd))
  {
     xed_uint_t disp_bits =
                xed_decoded_inst_get_memory_displacement_width( &xedd,0 );
     if (disp_bits)
     {
        *i_displacement = xed_decoded_inst_get_memory_displacement ( &xedd, 0 );
     }
  }

  unsigned noperands;
  const xed_inst_t* xi = xed_decoded_inst_inst(&xedd);
  xed_operand_t* op;
  const xed_operand_t* op0 = xed_inst_operand(xi,0);
  const xed_operand_t* op1 = xed_inst_operand(xi,1);
  const xed_operand_t* op2 = xed_inst_operand(xi,2);
  const xed_operand_t* op3 = xed_inst_operand(xi,3);
  xed_operand_action_enum_t rw;
  xed_operand_enum_t r, op_name;
  xed_uint_t bits;

  *i_vec_reg_number_0 = LIBXSMM_X86_VEC_REG_UNDEF;
  *i_vec_reg_number_1 = LIBXSMM_X86_VEC_REG_UNDEF;
  *i_vec_reg_number_2 = LIBXSMM_X86_VEC_REG_UNDEF;
  *i_is_store = 0;
  noperands = xed_inst_noperands(xi);
  i = noperands-1;
  while ( i >= 0 )
  {
    op = (xed_operand_t*) xed_inst_operand(xi,i);
    op_name = xed_operand_name ( op );
    r = xed_decoded_inst_get_reg ( &xedd, op_name );
    if ( (i == 0) && (r == XED_REG_INVALID) ) *i_is_store = 1;
    if ( op_name == XED_OPERAND_IMM0 ) {
       char buf1[64];
       const unsigned int no_leading_zeros=0;
       xed_uint_t ibits;
       const xed_bool_t lowercase = 1;
       ibits = xed_decoded_inst_get_immediate_width_bits(&xedd);
       if (xed_decoded_inst_get_immediate_is_signed(&xedd)) {
          xed_uint_t rbits = ibits?ibits:8;
          xed_int32_t x = xed_decoded_inst_get_signed_immediate(&xedd);
          xed_uint64_t y = XED_STATIC_CAST(xed_uint64_t,
                                           xed_sign_extend_arbitrary_to_64(
                                                       (xed_uint64_t)x,
                                                       ibits));
          xed_itoa_hex_ul(buf1, y, rbits, no_leading_zeros, 64, lowercase);
          *i_imm8 = y;
       } else {
          xed_uint64_t x =xed_decoded_inst_get_unsigned_immediate(&xedd);
          xed_uint_t rbits = ibits?ibits:16;
          xed_itoa_hex_ul(buf1, x, rbits, no_leading_zeros, 64, lowercase);
          *i_imm8 = x;
       }
    }
    if ( (r >= XED_REG_K0) && (r <= XED_REG_K7) ) {
       r -= XED_REG_K0;
       *i_mask_reg_number = (unsigned int) r;
    }
    if ( (r >= XED_REG_XMM0) && (r <= XED_REG_XMM31) ) {
       r -= XED_REG_XMM0;
       if ( *i_vec_reg_number_0 == LIBXSMM_X86_VEC_REG_UNDEF ) *i_vec_reg_number_0 = r;
       else if ( *i_vec_reg_number_1 == LIBXSMM_X86_VEC_REG_UNDEF ) *i_vec_reg_number_1 = r;
       else if ( *i_vec_reg_number_2 == LIBXSMM_X86_VEC_REG_UNDEF ) *i_vec_reg_number_2 = r;
       *i_vector_name = 'x';
    }
    if ( (r >= XED_REG_YMM0) && (r <= XED_REG_YMM31) ) {
       r -= XED_REG_YMM0;
       if ( *i_vec_reg_number_0 == LIBXSMM_X86_VEC_REG_UNDEF ) *i_vec_reg_number_0 = r;
       else if ( *i_vec_reg_number_1 == LIBXSMM_X86_VEC_REG_UNDEF ) *i_vec_reg_number_1 = r;
       else if ( *i_vec_reg_number_2 == LIBXSMM_X86_VEC_REG_UNDEF ) *i_vec_reg_number_2 = r;
       *i_vector_name = 'y';
    }
    if ( (r >= XED_REG_ZMM0) && (r <= XED_REG_ZMM31) ) {
       r -= XED_REG_ZMM0;
       if ( *i_vec_reg_number_0 == LIBXSMM_X86_VEC_REG_UNDEF ) *i_vec_reg_number_0 = r;
       else if ( *i_vec_reg_number_1 == LIBXSMM_X86_VEC_REG_UNDEF ) *i_vec_reg_number_1 = r;
       else if ( *i_vec_reg_number_2 == LIBXSMM_X86_VEC_REG_UNDEF ) *i_vec_reg_number_2 = r;
       *i_vector_name = 'z';
    }
    --i;
  }

  return 1;
}
#endif

#ifdef XED_DECODE_TESTING
/* Note: UNDEF is treated as 0 */
/* Note this returns 0 if all is well. >0 if there's a hard mismatch.
 * <0 if there's a soft mismatch. A soft mismatch is expecting i_vector_name='z' but it is 'x' instead. This is because libxsmm actually tolerates this kind of abuse. If you tell it i_vector_name='z' on a case that only works with xmm, libxsmm happily ignores your request. Is that a soft error or a hard one??? */
int xed_decode_mismatches_against_libxsmm ( char *xed_instr_name,
                                            char *instr_name,
                                            int twoops,
                                            int first,
                                            unsigned int xsmm_base,
                                            unsigned int xed_base,
                                            unsigned int xsmm_idx,
                                            unsigned int xed_idx,
                                            unsigned int xsmm_scale,
                                            unsigned int xed_scale,
                                            int xsmm_disp,
                                            int xed_disp,
                                            char xsmm_vector_name,
                                            char xed_vector_name,
                                            unsigned int xsmm_vec0,
                                            unsigned int xed_vec0,
                                            unsigned int xsmm_vec1,
                                            unsigned int xed_vec1,
                                            unsigned int xsmm_vec2,
                                            unsigned int xed_vec2,
                                            unsigned int xsmm_mask0,
                                            unsigned int xed_mask0,
                                            unsigned int xsmm_zerom,
                                            unsigned int xed_zerom,
                                            unsigned int xsmm_store,
                                            unsigned int xed_store,
                                            unsigned short xsmm_imm8,
                                            unsigned short xed_imm8 )
{
    int okay = 0;
    unsigned int lenx = strlen(xed_instr_name);

    if ( strncmp(instr_name,xed_instr_name,lenx) ) {
       printf("Decode problem in the name mismatching. Looking for:%s Got:%s\n",instr_name,xed_instr_name);
       return ( 1 );
    }
    if ( xsmm_base != xed_base ) {
       printf("Decode base mismatch (exp=%d,got=%d) at byte %d\n",xsmm_base,xed_base,first);
       return ( 2 );
    }
    if ( (xsmm_idx != xed_idx) && (xsmm_idx!=4) ) {
       printf("Decode idx mismatch (exp=%d,got=%d) at byte %d\n",xsmm_idx,xed_idx,first);
       return ( 3 );
    }
    if ( (xsmm_idx != LIBXSMM_X86_GP_REG_UNDEF) && (xsmm_idx != 4) && (xsmm_scale!=xed_scale) ) {
       printf("Decode i_scale mismatch (exp=%d,got=%d) at byte %d\n",xsmm_scale,xed_scale,first);
       return (4);
    }
    if ( xsmm_disp != xed_disp ) {
       printf("Decode i_disp mismatch (exp=%d,got=%d) at byte %d\n",xsmm_disp,xed_disp,first);
       return ( 5 ) ;
    }
    if ( xsmm_vector_name != xed_vector_name ) {
       printf("Decode i_vector_name mismatch (exp=%c,got=%c) at byte %d\n",xsmm_vector_name,xed_vector_name,first);
       return (  -6 ) ;
    }
    if ( xsmm_vec0 != xed_vec0 ) {
       /* Count UNDEF as 0 */
       okay = 0;
       if ( (xsmm_vec0==LIBXSMM_X86_VEC_REG_UNDEF) && (xed_vec0 == 0) ) okay= 1;
       if ( (xsmm_vec0==LIBXSMM_X86_VEC_REG_UNDEF) && (xsmm_vec1==LIBXSMM_X86_VEC_REG_UNDEF) && (xsmm_vec2==xed_vec0) && (xed_vec1==LIBXSMM_X86_VEC_REG_UNDEF) && (xed_vec2==LIBXSMM_X86_VEC_REG_UNDEF) ) okay = 1;
       if ( okay == 0 ) {
          printf("Decode i_vec_reg_number_0 mismatch (exp=%d,got=%d) at byte %d\n",xsmm_vec0,xed_vec0,first);
          printf("xsmm_vec0=%d xsmm_vec1=%d xsmm_vec2=%d\n",xsmm_vec0,xsmm_vec1,xsmm_vec2);
          printf("xed_vec0=%d xed_vec1=%d xed_vec2=%d\n",xed_vec0,xed_vec1,xed_vec2);
          return ( 7 );
       }
    }
    if ( xsmm_vec1 != xed_vec1 ) {
       if ( twoops == 1 ) {
          /* Don't mark something as wrong if it's UNDEF vs. 0 */
          if ( (xsmm_vec1 == LIBXSMM_X86_VEC_REG_UNDEF) && (xed_vec1==0) ) {
          } else {
             /* If LIBXSMM passes in 0, UNDEF, x and XED gets 0, x, UNDEF-> this is ok */
             if ( ((xsmm_vec1==LIBXSMM_X86_VEC_REG_UNDEF) || (xsmm_vec2==LIBXSMM_X86_VEC_REG_UNDEF)) && ( xsmm_vec1==xed_vec2 ) && (xsmm_vec2==xed_vec1) ) {
             } else {
                printf("Decode i_vec_reg_number_1 twoop mismatch (exp=%d,got=%d) at byte %d\n",xsmm_vec1,xed_vec1,first);
                printf("xsmm_vec0=%d xsmm_vec1=%d xsmm_vec2=%d\n",xsmm_vec0,xsmm_vec1,xsmm_vec2);
                printf("xed_vec0=%d xed_vec1=%d xed_vec2=%d\n",xed_vec0,xed_vec1,xed_vec2);
                return ( 8 );
             }
          }
       } else {
          if ( (xsmm_vec1 == LIBXSMM_X86_VEC_REG_UNDEF) && (xed_vec1==0) ) {
          } else {
             printf("Decode i_vec_reg_number_1 3-op mismatch (exp=%d,got=%d) at byte %d\n",xsmm_vec1,xed_vec1,first);
             return ( 9 );
          }
       }
    }
    if ( xsmm_vec2 != xed_vec2 ) {
       okay = 0;
       if ( (xsmm_vec2 == LIBXSMM_X86_VEC_REG_UNDEF) && (xed_vec2==0) ) okay= 1;
       if ( ((xsmm_vec1==LIBXSMM_X86_VEC_REG_UNDEF) || (xsmm_vec2==LIBXSMM_X86_VEC_REG_UNDEF)) && ( xsmm_vec1==xed_vec2 ) && (xsmm_vec2==xed_vec1) ) okay= 1;
       if ( (xsmm_vec0==LIBXSMM_X86_VEC_REG_UNDEF) && (xsmm_vec1==LIBXSMM_X86_VEC_REG_UNDEF) && (xsmm_vec2==xed_vec0) && (xed_vec1==LIBXSMM_X86_VEC_REG_UNDEF) && (xed_vec2==LIBXSMM_X86_VEC_REG_UNDEF) ) okay=1 ;
       if ( okay == 0 ) {
          printf("Decode i_vec_reg_number_2 mismatch (exp=%d,got=%d) at byte %d\n",xsmm_vec2,xed_vec2,first);
          printf("xsmm_vec0=%d xsmm_vec1=%d xsmm_vec2=%d\n",xsmm_vec0,xsmm_vec1,xsmm_vec2);
          printf("xed_vec0=%d xed_vec1=%d xed_vec2=%d\n",xed_vec0,xed_vec1,xed_vec2);
          printf("twoops=%d\n",twoops);
          return ( 9 );
       }
    }
    if ( xsmm_store != xed_store ) {
       printf("Decode i_is_store mismatch (exp=%d,got=%d) at byte %d\n",xsmm_store,xed_store,first);
       return ( 10 );
    }
    if ( (xsmm_imm8 != xed_imm8) && (xed_imm8!=LIBXSMM_X86_IMM_UNDEF) ) {
       printf("Decode imm8 mismatch (exp=%d,got=%d) at byte %d\n",xsmm_imm8, xed_imm8, first );
       return ( 11 );
    }

    return 0;
}
#endif
