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
#include "generator_transpose_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

/*
#define GENERATOR_TRANSPOSE_DEBUG
*/


/* d_ymm_or_zmm is automatically generated dispatching code
   Even on Skylake/KNL, zmm code doesn't always run better than ymm code.
   Given i_m x i_n matrix to transpose:
      This returns 1 if we should use xmm/ymm and not zmm
      This returns 2 if we should use zmm and not xmm/ymm
   i_m = number of rows of A to transpose
   i_n = number of columns of A to transpose
   i_avx512 (based on what the processor can do, not necessarily what's best)
    =0 to use AVX2 or below, 1 for zmm on Skylake, 2 for zmm on Knights Landing
*/
LIBXSMM_API_INLINE
int d_ymm_or_zmm(
         const int i_m,
         const int i_n,
         const int i_avx512
         )
{
  double dm, dn, dtmp;
  int l_retval;

  if ( !i_avx512 ) return 1;

  dm = (double) i_m;
  dn = (double) i_n;

  if ( i_avx512 == 1 )
  {
    /* Skylake dispatching logic */
    if ( dn <= 4.00000 )
    {
      dtmp = 1.00000;
    } else {
      if ( dn <= 12.00000 )
      {
        if ( dm <= 5.00000)
        {
          dtmp = 1.00000;
        } else {
          dtmp = 0.00916*dm - 0.16182*dn + 2.66904;
        }
      } else {
        dtmp = 0.02409*dm + 0.00486*dn + 1.25085;
      }
    }
  } else {
    /* Knights Landing dispatching logic */
    if ( -2.30000*dm + 2.00000*dn <= -6.00000 )
    {
      if ( dn <= 2.00000 )
      {
        dtmp = 1.00000;
      } else {
        if ( dn <= 4.00000 )
        {
          dtmp = 0.00032*dm - 0.69532*dn + 4.00575;
        } else {
          if ( -2.50000*dm - 1.50000*dn <= -32.00000)
          {
            if ( dm <= 17.00000 )
            {
              dtmp = -0.07867*dm - 0.01862*dn + 2.97591;
            } else {
              dtmp = 2.00000;
            }
          } else {
            dtmp = -0.40000*dm - 0.46667*dn + 7.20000;
          }
        }
      }
    } else {
      dtmp = 0.01791*dm + 0.00141*dn + 1.43536;
    }
  }
  /* Now turn it into an integer */
  l_retval = (int) dtmp;
  l_retval = LIBXSMM_MAX(l_retval,1);
  if ( dtmp - ((double) l_retval) >= 0.5 ) ++l_retval;
  l_retval = LIBXSMM_MIN(l_retval,2);
  return ( l_retval );
}


/* load_mask_into_var loads a ymm-based-mask based on the remainder m into
 *    ymm0 or ymm13 depending on "reg"
 * m = size of the border (should be less than the register size)
 * datasize= number of bytes for 1 unit (4 for single, 8 for double or complex)
 * reg = 0 or 13 to indicate which ymm register to use
 * buf = The buffer to contain the instruction/opcode sequence
 * loc = The location inside the buffer to store the new instruction bytes */
LIBXSMM_API_INLINE
void load_mask_into_var (
                          const int m,
                          const int datasize,
                          int reg,
                          unsigned char *buf,
                          int *loc
                        )
{
   unsigned char by=0;
   int i = *loc, m2, j;

   m2 = m;
   if ( datasize > 4 ) m2 = (datasize/4)*m;

   /* Currently, the transpose generator uses ymm0 and ymm13 only */
   if ( (reg != 0) && (reg != 13) )
   {
      fprintf(stderr,"strange register value into load_mask_into_var\n");
      exit(-1);
   }
   if ( m == 1 ) by = 0x80; else by = 0;
   buf[i]=0xeb; buf[i+1]=0x20; i=i+2; /* unconditional jmp past the data */
   for ( j = 1; j <= 8; j++ )
   {
      if ( m2 >= j ) by = 0; else by = 0x80;
      buf[i]=0x00; buf[i+1]=0x00; buf[i+2]=0x80; buf[i+3]=(unsigned char)(0xbf-by); i+=4;
   }
   /* The below is doing vmovups .data(%rip), %ymm(reg) */
   if ( reg == 0 )
   {
      buf[i]=0xc5; buf[i+1]=0xfc; buf[i+2]=0x10; buf[i+3]=0x05; i+=4;
   } else { /* reg == 13 */
      buf[i]=0xc5; buf[i+1]=0x7c; buf[i+2]=0x10; buf[i+3]=0x2d; i+=4;
   }
   buf[i]=0xd8; buf[i+1]=0xff; buf[i+2]=0xff; buf[i+3]=0xff; i+=4;

   *loc = i;
}


/* gen_one_trans generates a mxn single transposition of a subset of A into B *
*    (assuming m and n are less than the register size.) We also need to     *
*    know any offsets into A or B to know where to load.                     *
* m = number of rows of source A (should be less than the register size)     *
* n = number of columns of source A (should be less than the register size)  *
* ldb = In general, we assume that ldb is the original "n", however this     *
*    routine can be called multiple times, and the "n" here might be the     *
*    border. For instance if the register size is 8, and the original n is 9,*
*    then ldb=9 but during one call, n will be 8 and the other call, n will  *
*    be 1. So the only way to know the original "n" is to look at ldb...     *
* offsetA = offset in BYTES (not elements) to load for the first A           *
* offsetB = offset in BYTES (not elements) to store for the first B          *
* datasize = 4 for single, 8 for double or single complex                    *
* avx512=0 to use AVX2 or below, 1 for zmm on Skylake,                       *
*        2 for zmm on Knights Landing                                        *
* maskvar=value used for masking with ymm0 (in case we can reuse it,         *
*        otherwise, we must use ymm13). Obviously only valid when avx512==0  *
*                                                                            *
* Note: Assumes rdi = &A, rsi = lda*datasize, r8 = lda*datasize*3,           *
*               rbx =lda*datasize*5, rbp=lda*datasize*7, rdx = &B            *
* TODO: fix assumptions to match register mapping!                           */
LIBXSMM_API_INLINE void gen_one_trans(
  libxsmm_generated_code*                 io_generated_code,
  const libxsmm_transpose_gp_reg_mapping* i_gp_reg_mapping,
  const int m,
  const int n,
  const int ldb,
  const int offsetA,
  const int offsetB,
  const int datasize,
  const int avx512,
  const int maskvar)
{
  int i = io_generated_code->code_size;
  unsigned char reg;
  int shiftmult;
  int m_nonone_border = 0;
  int n_nonone_border = 0;
  int m_fits_in_xmmymm = 0;
  int n_fits_in_xmm = 0;
  int m_fits_in_xmm = 0;
  int m_fits_in_ymm = 0;
  unsigned int REGSIZE = 4;
  unsigned int l_instr;
  char cval = 'x';

  if (datasize == 8)
  {
    shiftmult = 8;
    if (m == 2) { cval = 'x'; }
    if (m == 4) { cval = 'y'; }
    if (m == 8) { cval = 'z'; }
    m_nonone_border = (m == 3);
    n_nonone_border = (n == 3);
    m_fits_in_xmmymm = ((m == 2) || (m == 4));
    n_fits_in_xmm = (n == 2);
    m_fits_in_xmm = (m == 2);
    m_fits_in_ymm = (m == 4);
    if (avx512) REGSIZE = 8; else REGSIZE = 4;
  }
  else {
    shiftmult = 4;
    if (m == 4) { cval = 'x'; }
    if (m == 8) { cval = 'y'; }
    m_nonone_border = ((m == 2) || (m == 3) || (m == 5) || (m == 6) || (m == 7));
    n_nonone_border = ((n == 2) || (n == 3) || (n == 5) || (n == 6) || (n == 7));
    m_fits_in_xmmymm = ((m == 4) || (m == 8));
    n_fits_in_xmm = (n == 4);
    m_fits_in_xmm = (m == 4);
    m_fits_in_ymm = (m == 8);
    if (avx512) REGSIZE = 16; else REGSIZE = 8;
  }

  /* Transposition has 3 parts: load the data, transpose it, store the data */
  /* The following is part 1: */
  if (!avx512)
  {
    if (m == 1)
    {
      if (datasize == 4)
      {
        l_instr = LIBXSMM_X86_INSTR_MOVL;
      }
      else {
        l_instr = LIBXSMM_X86_INSTR_MOVQ;
      }
      io_generated_code->code_size = i;
      /* Do instructions like: movl (%rdi), %r9d */
      if (n >= 1) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetA, i_gp_reg_mapping->gp_reg_n_loop, 0);
      /* movl (%rdi,%rsi,1), %r10d */
      if (n >= 2) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_lda, 1, offsetA, LIBXSMM_X86_GP_REG_R10, 0);
      /* movl (%rdi,%rsi,2), %r11d */
      if (n >= 3) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_lda, 2, offsetA, LIBXSMM_X86_GP_REG_R11, 0);
      /* movl (%rdi,%r8 ,1), %eax  */
      if (n >= 4) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_m_loop, 1, offsetA, LIBXSMM_X86_GP_REG_RAX, 0);
      /* movl (%rdi,%rsi,4), %r12d */
      if (n >= 5) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_lda, 4, offsetA, LIBXSMM_X86_GP_REG_R12, 0);
      /* movl (%rdi,%rbx,1), %r13d */
      if (n >= 6) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_RBX, 1, offsetA, LIBXSMM_X86_GP_REG_R13, 0);
      /* movl (%rdi,%r8 ,2), %r14d */
      if (n >= 7) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_m_loop, 2, offsetA, LIBXSMM_X86_GP_REG_R14, 0);
      /* movl (%rdi,%rbp,1), %r15d */
      if (n >= 8) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_RBP, 1, offsetA, LIBXSMM_X86_GP_REG_R15, 0);
      i = io_generated_code->code_size;
    } /* m==1 */
  }    /* !avx512 */

  if (!avx512)
  {
    reg = 0;
  }
  else {
    cval = 'z';
    if (m % REGSIZE == 0) reg = 0; else reg = 1;
  }
  if ((!avx512 && m_fits_in_xmmymm) || (avx512))
  {
    io_generated_code->code_size = i;
    /* Do instructions like: vmovups (%rdi), zmm1{%k1} */
    if (n>0) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetA, cval, 1, reg, 1, 0);
    /* vmovups (%rdi,%rsi,1), zmm2{%k1} */
    if (n>1) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_lda, 1, offsetA, cval, 2, reg, 1, 0);
    /* vmovups (%rdi,%rsi,2), zmm3{%k1} */
    if (n>2) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_lda, 2, offsetA, cval, 3, reg, 1, 0);
    /* vmovups (%rdi,%r8 ,1), zmm4{%k1} */
    if (n>3) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_m_loop, 1, offsetA, cval, 4, reg, 1, 0);
    /* vmovups (%rdi,%rsi,4), zmm5{%k1} */
    if (n>4) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_lda, 4, offsetA, cval, 5, reg, 1, 0);
    /* vmovups (%rdi,%rbx,1), zmm6{%k1} */
    if (n>5) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_RBX, 1, offsetA, cval, 6, reg, 1, 0);
    /* vmovups (%rdi,%r8 ,2), zmm7{%k1} */
    if (n>6) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_m_loop, 2, offsetA, cval, 7, reg, 1, 0);
    /* vmovups (%rdi,%rbp,1), zmm8{%k1} */
    if (n>7) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_RBP, 1, offsetA, cval, 8, reg, 1, 0);
    i = io_generated_code->code_size;
  }

  if (!avx512 && m_nonone_border)
  {
    /* We need a masked mov: vmaskmovps (%rdi), %ymm0, %ymm1 */
    if (datasize == 8) l_instr = LIBXSMM_X86_INSTR_VMASKMOVPD;
    else                 l_instr = LIBXSMM_X86_INSTR_VMASKMOVPS;
    io_generated_code->code_size = i;
    libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetA, 'y', 1, 0, 0);
    if (n>1) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_lda, 1, offsetA, 'y', 2, 0, 0);
    if (n>2) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_lda, 2, offsetA, 'y', 3, 0, 0);
    if (n>3) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_m_loop, 1, offsetA, 'y', 4, 0, 0);
    if (n>4) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_lda, 4, offsetA, 'y', 5, 0, 0);
    if (n>5) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_RBX, 1, offsetA, 'y', 6, 0, 0);
    if (n>6) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, i_gp_reg_mapping->gp_reg_m_loop, 2, offsetA, 'y', 7, 0, 0);
    if (n>7) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_a, LIBXSMM_X86_GP_REG_RBP, 1, offsetA, 'y', 8, 0, 0);
    i = io_generated_code->code_size;
  }
  /* Part 1 is done. The data is loaded */

  /* Transpose the data: */
  if (avx512 || (!avx512 && (m > 1) && (ldb>1)))
  {
    if (!avx512)
    {
      if (datasize == 8)
      {
        io_generated_code->code_size = i;
        /* vunpcklpd %ymm2, %ymm1, %ymm5 */
        libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKLPD, 'y', 2, 1, 5);
        /* vunpcklpd %ymm4, %ymm3, %ymm6 */
        libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKLPD, 'y', 4, 3, 6);
        /* vunpckhpd %ymm2, %ymm1, %ymm7 */
        if (m > 1) libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKHPD, 'y', 2, 1, 7);
        /* vunpckhpd %ymm4, %ymm3, %ymm7 */
        if (m > 1) libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKHPD, 'y', 4, 3, 8);
        if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 6, 5, 1, 32);
        if (m>1) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 8, 7, 2, 32);
        if (m>2) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 6, 5, 3, 49);
        if (m>3) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 8, 7, 4, 49);
        i = io_generated_code->code_size;
      }
      else {
        /* single precision: */
        io_generated_code->code_size = i;
        /* vunpcklps %ymm2, %ymm1, %ymm9 */
        libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKLPS, 'y', 2, 1, 9);
        /* vunpckhps %ymm2, %ymm1, %ymm1 */
        libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKHPS, 'y', 2, 1, 1);
        /* vunpcklps %ymm4, %ymm3, %ymm10 */
        libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKLPS, 'y', 4, 3, 10);
        /* vunpckhps %ymm4, %ymm3, %ymm2 */
        libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKHPS, 'y', 4, 3, 2);
        /* vunpcklps %ymm6, %ymm5, %ymm11 */
        libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKLPS, 'y', 6, 5, 11);
        /* vunpckhps %ymm6, %ymm5, %ymm3 */
        libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKHPS, 'y', 6, 5, 3);
        /* vunpcklps %ymm8, %ymm7, %ymm12 */
        libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKLPS, 'y', 8, 7, 12);
        /* vunpckhps %ymm8, %ymm7, %ymm4 */
        libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKHPS, 'y', 8, 7, 4);
        if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFPS, 'y', 10, 9, 5, 68);
        if (m>1) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFPS, 'y', 10, 9, 6, 238);
        if (m>2) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFPS, 'y', 2, 1, 7, 68);
        if (m>3) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFPS, 'y', 2, 1, 8, 238);
        if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFPS, 'y', 12, 11, 9, 68);
        if (m>1) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFPS, 'y', 12, 11, 10, 238);
        if (m>2) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFPS, 'y', 4, 3, 11, 68);
        if (m>3) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFPS, 'y', 4, 3, 12, 238);
        if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 9, 5, 1, 32);
        if (m>1) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 10, 6, 2, 32);
        if (m>2) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 11, 7, 3, 32);
        if (m>3) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 12, 8, 4, 32);
        if (m>4) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 9, 5, 5, 49);
        if (m>5) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 10, 6, 6, 49);
        if (m>6) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 11, 7, 7, 49);
        if (m>7) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VPERM2F128, 'y', 12, 8, 8, 49);
        i = io_generated_code->code_size;
      }
    }
    else { /* avx512 */
      /* vshuff64x2 $0xEE, %zmm3 , %zmm1 , %zmm9  */
      io_generated_code->code_size = i;
      if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 3, 1, 9, 0xEE);
      /* vshuff64x2 $0x44, %zmm3 , %zmm1 , %zmm1  */
      if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 3, 1, 1, 0x44);
      /* vshuff64x2 $0xEE, %zmm4 , %zmm2 , %zmm10 */
      if (m>2) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 4, 2, 10, 0xEE);
      /* vshuff64x2 $0x44, %zmm4 , %zmm2 , %zmm2  */
      if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 4, 2, 2, 0x44);
      /* vshuff64x2 $0xEE, %zmm7 , %zmm5 , %zmm11 */
      if (m>4) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 7, 5, 11, 0xEE);
      /* vshuff64x2 $0x44, %zmm7 , %zmm5 , %zmm3  */
      if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 7, 5, 3, 0x44);
      /* vshuff64x2 $0xEE, %zmm8 , %zmm6 , %zmm12 */
      if (m>4) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 8, 6, 12, 0xEE);
      /* vshuff64x2 $0x44, %zmm8 , %zmm6 , %zmm4  */
      if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 8, 6, 4, 0x44);
      /* vshuff64x2 $0xDD, %zmm3 , %zmm1 , %zmm6  */
      if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 3, 1, 6, 0xDD);
      /* vshuff64x2 $0x88, %zmm3 , %zmm1 , %zmm5  */
      if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 3, 1, 5, 0x88);
      /* vshuff64x2 $0xDD, %zmm11, %zmm9 , %zmm8  */
      if (m>6) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 11, 9, 8, 0xDD);
      /* vshuff64x2 $0x88, %zmm11, %zmm9 , %zmm7  */
      if (m>4) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 11, 9, 7, 0x88);
      /* vshuff64x2 $0x88, %zmm12, %zmm10, %zmm11 */
      if (m>4) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 12, 10, 11, 0x88);
      /* vshuff64x2 $0xDD, %zmm12, %zmm10, %zmm12 */
      if (m>6) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 12, 10, 12, 0xDD);
      /* vshuff64x2 $0xDD, %zmm4 , %zmm2 , %zmm10 */
      if (m>2) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 4, 2, 10, 0xDD);
      /* vshuff64x2 $0x88, %zmm4 , %zmm2 , %zmm9  */
      if (m>0) libxsmm_x86_instruction_vec_shuffle_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VSHUFF64X2, 'z', 4, 2, 9, 0x88);

      /* vunpcklpd  %zmm9 , %zmm5, %zmm1 */
      if (m>0) libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKLPD, 'z', 9, 5, 1);
      /* vunpckhpd  %zmm9 , %zmm5, %zmm2 */
      if (m>1) libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKHPD, 'z', 9, 5, 2);
      /* vunpcklpd  %zmm10, %zmm6, %zmm3 */
      if (m>2) libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKLPD, 'z', 10, 6, 3);
      /* vunpckhpd  %zmm10, %zmm6, %zmm4 */
      if (m>3) libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKHPD, 'z', 10, 6, 4);
      /* vunpcklpd  %zmm11, %zmm7, %zmm5 */
      if (m>4) libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKLPD, 'z', 11, 7, 5);
      /* vunpckhpd  %zmm11, %zmm7, %zmm6 */
      if (m>5) libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKHPD, 'z', 11, 7, 6);
      /* vunpcklpd  %zmm12, %zmm8, %zmm7 */
      if (m>6) libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKLPD, 'z', 12, 8, 7);
      /* vunpckhpd  %zmm12, %zmm8, %zmm8 */
      if (m>7) libxsmm_x86_instruction_vec_compute_reg(io_generated_code, LIBXSMM_X86_SSE3, LIBXSMM_X86_INSTR_VUNPCKHPD, 'z', 12, 8, 8);
      i = io_generated_code->code_size;
    }

    if (n_fits_in_xmm)
    {
      cval = 'x';
    }
    else {
      cval = 'y';
    }
  }

  if (!avx512)
  {
    /* Special case when ldb==1-> just do a copy */
    if (ldb == 1)
    {
      if (m_fits_in_xmm)
      {
        /* vmovups %xmm1, offsetB(%rdx) */
        io_generated_code->code_size = i;
        libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB, 'x', 1, 0, 0, 1);
        i = io_generated_code->code_size;
      }
      if (m_nonone_border)
      {
        io_generated_code->code_size = i;
        if (datasize == 8) l_instr = LIBXSMM_X86_INSTR_VMASKMOVPD;
        else                 l_instr = LIBXSMM_X86_INSTR_VMASKMOVPS;
        libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB, 'y', 1, 0, 1);
        i = io_generated_code->code_size;
      }
      if (m_fits_in_ymm)
      {
        /* vmovups %ymm1, (%rdx) */
        io_generated_code->code_size = i;
        libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB, 'y', 1, 0, 0, 1);
        i = io_generated_code->code_size;
      }
    }
  }

  /* Part 3: Store out the data */
  if (avx512 || (!avx512 && (m > 1) && (ldb > 1)))
  {
    if (!avx512 && ((n == 1) || n_nonone_border))
    {
      if (maskvar == n)
      {
        reg = 0;    /* ymm0 already contains the right mask */
      }
      else {
        reg = 13;   /* ymm13 better have the right mask */
      }
      io_generated_code->code_size = i;
      if (datasize == 8) l_instr = LIBXSMM_X86_INSTR_VMASKMOVPD;
      else                 l_instr = LIBXSMM_X86_INSTR_VMASKMOVPS;
      libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB, 'y', 1, reg, 1);
      if (m>1) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + shiftmult*ldb, 'y', 2, reg, 1);
      if (m>2) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 2 * shiftmult*ldb, 'y', 3, reg, 1);
      if (m>3) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 3 * shiftmult*ldb, 'y', 4, reg, 1);
      if (m>4) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 4 * shiftmult*ldb, 'y', 5, reg, 1);
      if (m>5) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 5 * shiftmult*ldb, 'y', 6, reg, 1);
      if (m>6) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 6 * shiftmult*ldb, 'y', 7, reg, 1);
      if (m>7) libxsmm_x86_instruction_vec_mask_move(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 7 * shiftmult*ldb, 'y', 8, reg, 1);
      i = io_generated_code->code_size;
    }
    else {
      if (!avx512)
      {
        io_generated_code->code_size = i;
        /* vmovups %ymm1, (%rdx) or xmm1 if cval=='x'*/
        libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB, cval, 1, 0, 0, 1);
        /* vmovups %ymm2, (%rdx) */
        if (m>1) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + shiftmult*ldb, cval, 2, 0, 0, 1);
        /* vmovups %ymm3, (%rdx) */
        if (m>2) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 2 * shiftmult*ldb, cval, 3, 0, 0, 1);
        /* vmovups %ymm4, (%rdx) */
        if (m>3) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 3 * shiftmult*ldb, cval, 4, 0, 0, 1);
        /* vmovups %ymm5, (%rdx) */
        if (m>4) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 4 * shiftmult*ldb, cval, 5, 0, 0, 1);
        /* vmovups %ymm6, (%rdx) */
        if (m>5) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 5 * shiftmult*ldb, cval, 6, 0, 0, 1);
        /* vmovups %ymm7, (%rdx) */
        if (m>6) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 6 * shiftmult*ldb, cval, 7, 0, 0, 1);
        /* vmovups %ymm8, (%rdx) */
        if (m>7) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 7 * shiftmult*ldb, cval, 8, 0, 0, 1);
        i = io_generated_code->code_size;
      }
      else { /* avx512 */
        cval = 'z';
        if (n % REGSIZE == 0) reg = 0; else reg = 2;
        io_generated_code->code_size = i;
        /* vmovups %zmm1, offsetB(%rdx) {%k2} (reg is the mask reg) */
        if (m>0) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB, cval, 1, reg, 0, 1);
        /* vmovups %zmm2, *(%rdx) {%k2} */
        if (m>1) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 8 * ldb, cval, 2, reg, 0, 1);
        if (m>2) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 16 * ldb, cval, 3, reg, 0, 1);
        if (m>3) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 24 * ldb, cval, 4, reg, 0, 1);
        if (m>4) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 32 * ldb, cval, 5, reg, 0, 1);
        if (m>5) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 40 * ldb, cval, 6, reg, 0, 1);
        if (m>6) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 48 * ldb, cval, 7, reg, 0, 1);
        if (m>7) libxsmm_x86_instruction_vec_move(io_generated_code, LIBXSMM_X86_AVX, LIBXSMM_X86_INSTR_VMOVUPS, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + 56 * ldb, cval, 8, reg, 0, 1);
        i = io_generated_code->code_size;
      } /* avx512 */
    }
  }

  if (!avx512)
  {
    if (m == 1)
    {
      io_generated_code->code_size = i;
      if (datasize == 4)
      {
        l_instr = LIBXSMM_X86_INSTR_MOVL;
      }
      else {
        l_instr = LIBXSMM_X86_INSTR_MOVQ;
      }
      /* movl %r9d, (%rdx) */
      if (n >= 1) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB, i_gp_reg_mapping->gp_reg_n_loop, 1);
      /* movl %r10d, (%rdx) */
      if (n >= 2) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + shiftmult, LIBXSMM_X86_GP_REG_R10, 1);
      /* movl %r11d, (%rdx) */
      if (n >= 3) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + shiftmult * 2, LIBXSMM_X86_GP_REG_R11, 1);
      /* movl %eax,  (%rdx) */
      if (n >= 4) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + shiftmult * 3, LIBXSMM_X86_GP_REG_RAX, 1);
      /* movl %r12d, (%rdx) */
      if (n >= 5) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + shiftmult * 4, LIBXSMM_X86_GP_REG_R12, 1);
      /* movl %r13d, (%rdx) */
      if (n >= 6) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + shiftmult * 5, LIBXSMM_X86_GP_REG_R13, 1);
      /* movl %r14d, (%rdx) */
      if (n >= 7) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + shiftmult * 6, LIBXSMM_X86_GP_REG_R14, 1);
      /* movl %r15d, (%rdx) */
      if (n >= 8) libxsmm_x86_instruction_alu_mem(io_generated_code, l_instr, i_gp_reg_mapping->gp_reg_b, LIBXSMM_X86_GP_REG_UNDEF, 1, offsetB + shiftmult * 7, LIBXSMM_X86_GP_REG_R15, 1);
      i = io_generated_code->code_size;
    }
  } /* avx512 */

    /* *loc = i; */
  io_generated_code->code_size = i;
}


LIBXSMM_API_INTERN
void libxsmm_generator_transpose_avx_avx512_kernel(
                libxsmm_generated_code*         io_generated_code,
                const libxsmm_trans_descriptor* i_trans_desc,
                int                             i_arch )
{
  libxsmm_transpose_gp_reg_mapping l_gp_reg_mapping;
  libxsmm_loop_label_tracker l_loop_label_tracker;

  const char *const cpuid = libxsmm_cpuid_name( i_arch );
  /* avx512 just represents whether we want to use zmm registers or not     *
   *      A value of 0 says not, a value of 1 targets AVX512_CORE, a value  *
   *      of 2 targets AVX512_MIC                                           */
  int avx512;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define gp register mapping */
  memset(&l_gp_reg_mapping, 0, sizeof(l_gp_reg_mapping));
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_R9;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_RSI;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_lda = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_ldb = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_m_loop = LIBXSMM_X86_GP_REG_R8;
  l_gp_reg_mapping.gp_reg_n_loop = LIBXSMM_X86_GP_REG_R9;
#endif
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;
  /* Actually, the logic is this: we need a, lda, and b. We don't need ldb  *
   * If n>=6, we need rbx                                                   *
   * If n>=8, we need rbp                                                   *
   * If LIBXSMM_MIN(n,REGSIZE)>=5 and m%REGSIZE==1, we need r12             *
   * If LIBXSMM_MIN(n,REGSIZE)>=6 and m%REGSIZE==1, we need r13             *
   * If LIBXSMM_MIN(n,REGSIZE)>=7 and m%REGSIZE==1, we need r14             *
   * If LIBXSMM_MIN(n,REGSIZE)>=8 and m%REGSIZE==1, we need r15             *
   * Otherwise, we get by with registers that don't require pushing/popping */

  /* define transposition kernel config */
  if (LIBXSMM_X86_AVX512_CORE <= i_arch) {
    avx512 = 1;
  } else if (LIBXSMM_X86_AVX512 <= i_arch) {
    avx512 = 2;
  } else if (LIBXSMM_X86_AVX <= i_arch) {
    avx512 = 0;
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }
  /* @Greg add more fields here */

  /* open asm */
  /* Note: I'm not sure exactly what to add to open_stream_transpose... this
   *    is for the regular assembly coding?                                  */
  libxsmm_x86_instruction_open_stream_transpose( io_generated_code, l_gp_reg_mapping.gp_reg_a,
                                                 l_gp_reg_mapping.gp_reg_lda, l_gp_reg_mapping.gp_reg_b,
                                                 l_gp_reg_mapping.gp_reg_ldb, cpuid );

  if ( io_generated_code->code_type > 1 )
  {
     unsigned char *buf = (unsigned char *) io_generated_code->generated_code;
     int i = io_generated_code->code_size;
     unsigned int m = i_trans_desc->m;
     unsigned int n = i_trans_desc->n;
     int loopm = 0, loopn= 0, mjmp, njmp;
     int imask = 0;
     int offsetA, offsetB, oldB;
     int j, k, m0, n0, shiftvalue, shiftmult;
     /* Note: the transpose routine only works when ldb is fixed at a value
      * So why do we need a variable "ldb"? Well, it's to keep track of
      * the original "n" and that's the only reason */
     unsigned int ldo = i_trans_desc->ldo;
     unsigned int ldb;
     /* REGSIZE is used for masking. REGSIZE is just:
      *           4 for double on ymm (unless m=1, then it's 8),
      *           8 for single on ymm or double on zmm,
      *           16 for single on zmm */
     unsigned int REGSIZE;
     int maskvar = 0;
     int datasize = i_trans_desc->typesize;

     if ( ldo < n )
     {
        /* This means that we didn't store ldb correctly. Not sure why, Greg
           thinks we should change/fix this. */
        ldb = n;
     } else {
        ldb = ldo;
     }

#ifdef GENERATOR_TRANSPOSE_DEBUG
     const unsigned int l_maxsize = io_generated_code->buffer_size;
     printf("Entering libxsmm_generator_transpose_avx_avx512_kernel with i loc=%d m=%d n=%d datasize=%d\n",i,m,n,datasize);
     printf("Space available: %d - %d = %d\n",l_maxsize,i,l_maxsize-i);
#endif
     assert(0 < datasize);
     if ( (datasize != 4) && (datasize != 8) )
     {
        fprintf(stderr,"Expecting a datasize of 4 or 8, but got %d\n",datasize);
        exit(-1);
     }
     /* Comment this next conditional out to *FORCE* AVX-512 dispatching */
     if ( avx512 )
     {
        /* Determine if we should really use ZMM registers, or not */
        if ( d_ymm_or_zmm( m, n, avx512 ) == 1 )
        {
           avx512 = 0; /* Ymm might be faster than zmm */
        }
        if ( datasize == 4 ) avx512 = 0; /* Don't use avx512 on real*4 */
     }
     if ( datasize == 8 )
     {
        shiftvalue = 3;
        shiftmult = 8;
        if ( avx512 ) { REGSIZE = 8; } else {
           if ( m == 1 ) REGSIZE = 8; else REGSIZE = 4;
        }
     } else {
        shiftvalue = 2;
        shiftmult = 4;
        if ( avx512 ) REGSIZE = 16; else REGSIZE = 8;
     }

     if ( avx512 )
     {
        m0 = m % REGSIZE;
        n0 = n % REGSIZE;
        if ( m0 > 0 )
        {
           k = m0;
           if ( k == 1 ) imask = 3;
           else if ( k == 2 ) imask = 15;
           else if ( k == 3 ) imask = 63;
           else if ( k == 4 ) imask = 255;
           else if ( k == 5 ) imask = 1023;
           else if ( k == 6 ) imask = 4095;
           else if ( k == 7 ) imask = 16383;
           /* movq imask, %r8: */
           io_generated_code->code_size = i;
           libxsmm_x86_instruction_alu_imm ( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, 8, imask );
           /* kmovw %r8d, %k1: */
           libxsmm_x86_instruction_mask_move ( io_generated_code, LIBXSMM_X86_INSTR_KMOVW, 8, 1 );
           i = io_generated_code->code_size;
        }
        if ( n0 > 0 )
        {
           k = n0;
           if ( k == 1 ) imask = 3;
           else if ( k == 2 ) imask = 15;
           else if ( k == 3 ) imask = 63;
           else if ( k == 4 ) imask = 255;
           else if ( k == 5 ) imask = 1023;
           else if ( k == 6 ) imask = 4095;
           else if ( k == 7 ) imask = 16383;
           /* movq imask, %r8: */
           io_generated_code->code_size = i;
           libxsmm_x86_instruction_alu_imm ( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, 8, imask );
           /* kmovw %r8d, %k2: */
           libxsmm_x86_instruction_mask_move ( io_generated_code, LIBXSMM_X86_INSTR_KMOVW, 8, 2 );
           i = io_generated_code->code_size;
        }
     }

     if ( n > 1 )
     {
        /* movslq (%rsi), %rsi   and salq $shiftvalue, %rsi */
        io_generated_code->code_size = i;
        libxsmm_x86_instruction_alu_mem ( io_generated_code, LIBXSMM_X86_INSTR_MOVSLQ, LIBXSMM_X86_GP_REG_RSI, LIBXSMM_X86_GP_REG_UNDEF, 1, 0, LIBXSMM_X86_GP_REG_RSI, 0 );
        libxsmm_x86_instruction_alu_imm ( io_generated_code, LIBXSMM_X86_INSTR_SALQ, LIBXSMM_X86_GP_REG_RSI, shiftvalue );
        i = io_generated_code->code_size;
     }
     if ( n >= 4 )
     {
        /* movq %rsi, %r8 and imul $3, %r8: */
        io_generated_code->code_size = i;
        libxsmm_x86_instruction_alu_reg ( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSI, LIBXSMM_X86_GP_REG_R8 );
        libxsmm_x86_instruction_alu_imm ( io_generated_code, LIBXSMM_X86_INSTR_IMUL, LIBXSMM_X86_GP_REG_R8, 3 );
        i = io_generated_code->code_size;
        if ( LIBXSMM_MIN(n,REGSIZE) >= 6 )
        {
           /* movq %rsi, %rbx and imul $5, %rbx : */
           io_generated_code->code_size = i;
           libxsmm_x86_instruction_alu_reg ( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSI, LIBXSMM_X86_GP_REG_RBX );
           libxsmm_x86_instruction_alu_imm ( io_generated_code, LIBXSMM_X86_INSTR_IMUL, LIBXSMM_X86_GP_REG_RBX, 5 );
           i = io_generated_code->code_size;
        }
        if ( LIBXSMM_MIN(n,REGSIZE) >= 8 )
        {
           /* movq %rsi, %rbp and imul $7, %rbp: */
           io_generated_code->code_size = i;
           libxsmm_x86_instruction_alu_reg ( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_RSI, LIBXSMM_X86_GP_REG_RBP );
           libxsmm_x86_instruction_alu_imm ( io_generated_code, LIBXSMM_X86_INSTR_IMUL, LIBXSMM_X86_GP_REG_RBP, 7 );
           i = io_generated_code->code_size;
        }
     }
#ifdef GENERATOR_TRANSPOSE_DEBUG
     printf("loc1 m=%d n=%d i=%d datasize=%d\n",m,n,i,datasize);
#endif

     /* Load any necessary masks into ymm0 and/or ymm13 */
     if ( !avx512 && (m != 1) )
     {
        int mt = m%REGSIZE;
        int nt = n%REGSIZE;
        int reg = 0;
        if ( datasize == 8 )
        {
           if ( mt == 3 )
           {
              io_generated_code->code_size = i;
              load_mask_into_var ( mt, datasize, reg, buf, &i );
              io_generated_code->code_size = i;
              if ( reg == 0 ) maskvar = mt;
              reg = 13;
           }
           if ( (nt == 1) || (nt == 3) )
           {
              io_generated_code->code_size = i;
              load_mask_into_var ( nt, datasize, reg, buf, &i );
              io_generated_code->code_size = i;
              if ( reg == 0 ) maskvar = nt;
           }
        } else if ( datasize == 4 )
        {
           if ( (mt==2) || (mt==3) || (mt==5) || (mt==6) || (mt==7) )
           {
              io_generated_code->code_size = i;
              load_mask_into_var ( mt, datasize, reg, buf, &i );
              io_generated_code->code_size = i;
              if ( reg == 0 ) maskvar = mt;
              reg = 13;
           }
           if ( (nt==1) || ((nt != mt) && (nt != 4)) )
           {
              io_generated_code->code_size = i;
              load_mask_into_var ( nt, datasize, reg, buf, &i );
              io_generated_code->code_size = i;
              if ( reg == 0 ) maskvar = nt;
           }
        }
     }

     /* Determine whether to use loops or not */
     if ( (n / REGSIZE) >= 2 ) loopn = n / REGSIZE; else loopn = 0;
     if ( (m / REGSIZE) >= 2 ) loopm = m / REGSIZE; else loopm = 0;

     /* To prevent and disable looping, just set loopm and loopn to 0 */
#ifdef PREVENT_TRANSPOSE_LOOPING
     loopm = 0;
     loopn = 0;
#endif

     if ( loopn > 0 ) {
        io_generated_code->code_size = i;
        libxsmm_x86_instruction_alu_imm ( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, l_gp_reg_mapping.gp_reg_ldb, loopn );
        libxsmm_x86_instruction_register_jump_back_label ( io_generated_code,  &l_loop_label_tracker );
        i = io_generated_code->code_size;
     }

     /* Here is the main loop and it's logic is simple. We just "stamp" a bunch
      * of smaller transpositions using the routine "get_one_trans()".
      * Eventually, incorporate loops into this for smaller footprints */
     offsetA = 0;
     offsetB = 0;
     oldB = 0;
     njmp = REGSIZE;
     if ( loopn > 0 ) njmp = REGSIZE*loopn;
     mjmp = REGSIZE;
     if ( loopm > 0 ) mjmp = REGSIZE*loopm;

     for (j = 1; j <= (int)n; j += njmp )
     {
        offsetA = 0;
        oldB = offsetB;

        if ( loopm > 0 ) {
           io_generated_code->code_size = i;
           libxsmm_x86_instruction_alu_imm ( io_generated_code, LIBXSMM_X86_INSTR_MOVQ, LIBXSMM_X86_GP_REG_R15, loopm );
           libxsmm_x86_instruction_register_jump_back_label ( io_generated_code,  &l_loop_label_tracker );
           i = io_generated_code->code_size;
        }

        for ( k = 1; k <= (int)m; k += mjmp )
        {
           io_generated_code->code_size = i;
           /* Note that the m, n parameters aren't the original m, n;
              which is why we also pass in this phony "ldb". Make certain this
              routine is never called with values in excess of REGSIZE */
#ifdef GENERATOR_TRANSPOSE_DEBUG
           printf("calling gen_one_trans mxn=%dx%d using %dx%d offsetA=%d offsetB=%d i=%d datasize=%d maskvar=%d\n",m,n,LIBXSMM_MIN(REGSIZE,((int)m)-k+1),LIBXSMM_MIN(REGSIZE,((int)n)-j+1),offsetA,offsetB,i,datasize,maskvar);
#endif
           /* This routine just does a single transpose at a time. */
           assert(k <= (int)(m + 1) && j <= (int)(n + 1));
           gen_one_trans(io_generated_code, &l_gp_reg_mapping,
                         LIBXSMM_MIN(REGSIZE,m-k+1),
                         LIBXSMM_MIN(REGSIZE,n-j+1),
                         ldb, offsetA, offsetB, datasize,
                         avx512, maskvar);
           if (0 != io_generated_code->last_error) return;
           i = io_generated_code->code_size;
#ifdef GENERATOR_TRANSPOSE_DEBUG
           printf("done calling gen_one_trans mxn=%dx%d using %dx%d offsetA=%d offsetB=%d i=%d datasize=%d maskvar=%d\n",m,n,LIBXSMM_MIN(REGSIZE,((int)m)-k+1),LIBXSMM_MIN(REGSIZE,((int)n)-j+1),offsetA,offsetB,i,datasize,maskvar);
#endif

           if ( loopm == 0 ) {
              offsetA += shiftmult*mjmp;
              offsetB += shiftmult*mjmp*ldb;
           } else if ( k==1 ) {
              io_generated_code->code_size = i;
              libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, l_gp_reg_mapping.gp_reg_a, shiftmult*REGSIZE );
              libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, l_gp_reg_mapping.gp_reg_b, shiftmult*REGSIZE*ldb );
              libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, LIBXSMM_X86_GP_REG_R15, 1 );
              libxsmm_x86_instruction_jump_back_to_label( io_generated_code, LIBXSMM_X86_INSTR_JG, &l_loop_label_tracker );
              i = io_generated_code->code_size;
           }
        }

        if ( loopm > 0 ) {
           io_generated_code->code_size = i;
           libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, l_gp_reg_mapping.gp_reg_b, shiftmult*REGSIZE*ldb*loopm );
           libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, l_gp_reg_mapping.gp_reg_a, shiftmult*REGSIZE*loopm );
           i = io_generated_code->code_size;
        }

        if ( j+REGSIZE <= n )
        {
           io_generated_code->code_size = i;
           /* addq %r8, %rdi: */
           if ( REGSIZE == 4 ) libxsmm_x86_instruction_alu_reg ( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_R8,  LIBXSMM_X86_GP_REG_RDI );
           /* addq %rbp, %rdi: */
           if ( REGSIZE == 8 ) libxsmm_x86_instruction_alu_reg ( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RBP, LIBXSMM_X86_GP_REG_RDI );
           /* addq %rsi, %rdi: */
           libxsmm_x86_instruction_alu_reg ( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, LIBXSMM_X86_GP_REG_RSI, LIBXSMM_X86_GP_REG_RDI );
           i = io_generated_code->code_size;
        }
        offsetB = oldB + shiftmult*REGSIZE;
     }

     if ( loopn > 0 ) {
        io_generated_code->code_size = i;
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_ADDQ, l_gp_reg_mapping.gp_reg_b, shiftmult*REGSIZE );
        libxsmm_x86_instruction_alu_imm( io_generated_code, LIBXSMM_X86_INSTR_SUBQ, l_gp_reg_mapping.gp_reg_ldb, 1 );
        libxsmm_x86_instruction_jump_back_to_label( io_generated_code, LIBXSMM_X86_INSTR_JG, &l_loop_label_tracker );
        i = io_generated_code->code_size;
     }

     io_generated_code->code_size = i;
#ifdef GENERATOR_TRANSPOSE_DEBUG
  printf("almost done with m=%d n=%d i=%d datasize=%d\n",m,n,i,datasize);
#endif
  }

  /* close asm: note that we really didn't need to push everything */
  libxsmm_x86_instruction_close_stream_transpose( io_generated_code, cpuid );
#ifdef GENERATOR_TRANSPOSE_DEBUG
  printf("done with m=%d n=%d i=%d\n",i_trans_desc->m,i_trans_desc->n,io_generated_code->code_size);
#endif
}

