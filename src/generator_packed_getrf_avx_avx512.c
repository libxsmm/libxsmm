/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Greg Henry, Hans Pabst, Timothy Costa (Intel Corp.)
******************************************************************************/
#include "generator_packed_getrf_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_packed_aux.h"
#include "generator_packed_gemmnn.h"
#include "generator_common.h"
#include "libxsmm_main.h"

#if 0
# define GENERATOR_PACKED_GETRF_DEBUG
#endif


LIBXSMM_API_INTERN
void libxsmm_generator_packed_getrf_avx_avx512_kernel( libxsmm_generated_code*        io_code,
                                                       const libxsmm_getrf_descriptor* i_packed_getrf_desc,
                                                       const char*                    i_arch )
{
  unsigned char *const buf = (unsigned char *) io_code->generated_code;
  libxsmm_loop_label_tracker l_loop_label_tracker /*= { 0 }*/;
  /* avx512 just represents whether we want to use zmm registers or not     *
   *      A value of 0 says not, a value of 1 targets AVX512_CORE, a value  *
   *      of 2 targets AVX512_MIC                                           */
  int avx512;
#if defined(_WIN32) || defined(__CYGWIN__)
  int l_matrix_gpreg = LIBXSMM_X86_GP_REG_RCX;
#else
  int l_matrix_gpreg = LIBXSMM_X86_GP_REG_RDI;
#endif
#if 0 /* TOD: introduce/use register mapping rather than directly/hard-coding registers */
  /* Just reuse transpose gp mapping */
  libxsmm_getrf_gp_reg_mapping l_gp_reg_mapping = { 0/*avoid warning "maybe used uninitialized" */ };
  /* define gp register mapping */
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
#endif
  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define transposition kernel config */
  if (strcmp(i_arch, "skx") == 0) {
    avx512 = 1;
  } else if (strcmp(i_arch, "knl") == 0 || strcmp(i_arch, "knm") == 0) {
    avx512 = 2;
  } else if (strcmp(i_arch, "snb") == 0 || strcmp(i_arch, "hsw") == 0) {
    avx512 = 0;
  } else {
    LIBXSMM_HANDLE_ERROR( io_code, LIBXSMM_ERR_UNSUP_ARCH );
    return;
  }
  /* @Greg add more fields here */

  /* @Greg add generator code here, please use functions defined in generator_x86_instructions.h */
  /* Todo-> I first want this code to work, and verify it works, then I can
   *        convert one instruction at a time to those in
   *        generator_x86_instructions.h. Or add to the existing instructions */

  if ( io_code->code_type > 1 )
  {
     unsigned int i = io_code->code_size;
     unsigned int m = i_packed_getrf_desc->m;
     unsigned int n = i_packed_getrf_desc->n;
     unsigned int lda = i_packed_getrf_desc->lda;
     const unsigned int lay = (unsigned int)i_packed_getrf_desc->layout;
     unsigned int datasz = (unsigned int)i_packed_getrf_desc->typesize;
     /*const double beta = 1.0;*/
     unsigned int m1=m, n1=n, mn;
     unsigned int j, k, ii;
     unsigned int tra=0, trb=0, trc=0, iunroll=3, junroll=3, loopi=1, loopj=1;
     /*int REGSIZE;*/
     int numb = 0;
     unsigned int bot, fincol;
     /*int nounit=0;*/
     unsigned int /*mb,*/ nb;
     /*int iun, jun;*/
     char regset = 'y';
     double one = 1.0;
     double none = -1.0;

     /* Register mapping: */
     int a0 = 0, a1 = 1, a2 = 2;
     int b0 = 3/*, b1 = 4, b2 = 5, b3*/;
     /*int c00 = 6, c01 = 7, c02 = 8, c03;*/
     /*int c10 = 9, c11 = 10, c12 = 11, c13;*/
     /*int c20 = 12, c21 = 13, c22 = 14, c23;*/
     /*int c30, c31, c32, c33;*/
     /*int c40, c41, c42, c43;*/
     /*int c0, c2, c3, c4;*/

     int onereg = 15;

     if ( lay == 101 )
     {
#if 0
        if (i_packed_trsm_desc->side == 'L' || i_packed_trsm_desc->side == 'l' ) side = 'R';
        else side = 'L';
        if (i_packed_trsm_desc->uplo == 'L' || i_packed_trsm_desc->uplo == 'l' ) uplo = 'U';
        else uplo = 'L';
        m1 = n; n1 = m;
#endif
        tra = 1; trb = 1; trc = 1;
     }
#if defined(GENERATOR_PACKED_GETRF_DEBUG)
printf("Inside libxsmm_generator_packed_getrf_avx_avx512_kernel: m=%d n=%d lay=%d lda=%d datasz=%d\n",m,n,lay,lda,datasz);
#endif
     if ( ( datasz !=4 ) && (datasz != 8) )
     {
        fprintf(stderr,"Expecting a datasize of 4 or 8 but got %u\n",datasz);
        exit(-1);
     }
     if ( avx512 < 0 || avx512 > 2 )
     {
        fprintf(stderr,"Expecting an integer between 0 and 2 for avx512, got %i\n",avx512);
        exit(-1);
     }
     if ( datasz == 4 && avx512 == 0 )
     {
        numb = 8;
        regset = 'y';
     } else if ( datasz == 8 && avx512 == 0 )
     {
        numb = 4;
        regset = 'y';
     } else if ( datasz == 4 && avx512 > 0 )
     {
        numb = 16;
        regset = 'z';
        iunroll = 4;
        junroll = 4;
        onereg = 25;
     } else if ( datasz == 8 && avx512 > 0 )
     {
        numb = 8;
        regset = 'z';
        iunroll = 4;
        junroll = 4;
        onereg = 25;
     }

     /* Determine ideal blocksizes: */
     nb = 2;
     if ( m1 <= 3 ) nb = 1;
     if ( n1 <= 2 ) nb = 1;
     mn = LIBXSMM_MIN(m1,n1);
     if ( mn >= 6 ) nb = 3;
     if ( mn >= 12 ) nb = 4;

     compact_set_one_ ( io_code, onereg, numb, datasz, regset );
#if 0
compact_store_matrix_gen_ ( io_code, tra, lda, 1, 1, onereg, numb, datasz, regset, l_matrix_gpreg );
mn=0;
#endif

     for ( ii = 1; ii <= mn; ii += nb ) {
        bot = LIBXSMM_MIN(ii+nb-1,mn);

        for ( j = ii; j <= bot; j++ ) {
           for ( i = j+1; i <= m1; i++ ) {
              if ( i == j+1 ) {
                 compact_load_matrix_gen_ ( io_code, tra, lda, j, j, a0, numb, datasz, regset, l_matrix_gpreg );
                 compact_divide_two_nums_ ( io_code, onereg, a0, a0, numb, regset );
              }
              compact_load_matrix_gen_ ( io_code, tra, lda, i, j, a1, numb, datasz, regset, l_matrix_gpreg );
              compact_mult_two_nums_ ( io_code, a0, a1, a1, numb, regset );
              fincol = bot;
              if ( i <= bot ) fincol = n1;
              for ( k = j+1; k <= fincol; k++ ) {
                 compact_load_matrix_gen_ ( io_code, tra, lda, i, k, a2, numb, datasz, regset, l_matrix_gpreg );
                 compact_load_matrix_gen_ ( io_code, tra, lda, j, k, b0, numb, datasz, regset, l_matrix_gpreg );
                 compact_fms_cminusab_ ( io_code, a2, a1, b0, numb, regset );
                 compact_store_matrix_gen_ ( io_code, tra, lda, i, k, a2, numb, datasz, regset, l_matrix_gpreg );

              }
              compact_store_matrix_gen_ ( io_code, tra, lda, i, j, a1, numb, datasz, regset, l_matrix_gpreg );
           }
        }
        if ( (bot < m1) && (bot < n1) ) {
/*
 *       Solve bottom right A22 part with a DGEMM("Notrans","Notrans",m-bot,n-bot,bot-ii+1,-1.0,A(bot+1,ii),lda,A(ii,bot+1),lda,1.0,A(bot+1,bot+1),lda)
 *       A(bot+1:m,bot+1:n) = A(bot+1:m,bot+1:n) - A(bot+1:m,ii:bot)*A(ii:bot,bot+1:n);
 *       */
           compact_gemmnn_(tra,trb,trc,bot+1,m1,ii,bot,ii,bot,bot+1,n1,bot+1,m1,bot+1,n1,none,l_matrix_gpreg,lda,l_matrix_gpreg,lda,one,l_matrix_gpreg,lda,io_code,numb,regset,iunroll,junroll,loopi,loopj);
        }      /* Nonempty DGEMM conditional */
     }        /* Main loop for LU */

  }

  { int i = io_code->code_size;
    buf[i++] = 0xc3; /* retq */
    io_code->code_size = i;
  }
  /*  close asm: note that we really didn't need to push everything */
/*
  libxsmm_x86_instruction_close_stream_transpose( io_code, i_arch );
*/

#if 0
#define DEBUG_GIVE_BYTE_CODE_OUTPUT
#endif
#ifdef DEBUG_GIVE_BYTE_CODE_OUTPUT
  buf = (unsigned char *) io_code->generated_code;
  printf("#Final Routine: \n");
  for ( i = 0; i < io_code->code_size; i+=8 ) {
    printf("#\tBytes %d-%d\n",i,i+7);
    printf(".byte 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x, 0x%02x\n",buf[i],buf[i+1],buf[i+2],buf[i+3],buf[i+4],buf[i+5],buf[i+6],buf[i+7]);
  }
#endif

}
