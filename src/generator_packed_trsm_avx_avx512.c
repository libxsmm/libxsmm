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
#include "generator_packed_trsm_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_packed_aux.h"
#include "generator_common.h"
#include "libxsmm_main.h"

#if 0
# define GENERATOR_PACKED_TRSM_DEBUG
#endif


LIBXSMM_API_INTERN
void libxsmm_generator_packed_trsm_avx_avx512_kernel( libxsmm_generated_code*        io_code,
                                                      const libxsmm_trsm_descriptor* i_packed_trsm_desc,
                                                      const char*                    i_arch )
{
  unsigned char *const buf = (unsigned char *) io_code->generated_code;
  libxsmm_loop_label_tracker l_loop_label_tracker /*= { 0 }*/;
  /* avx512 just represents whether we want to use zmm registers or not     *
   *      A value of 0 says not, a value of 1 targets AVX512_CORE, a value  *
   *      of 2 targets AVX512_MIC                                           */
  int avx512;
#if 0 /* TOD: introduce/use register mapping rather than directly/hard-coding registers */
  /* Just reuse transpose gp mapping */
  libxsmm_trsm_gp_reg_mapping l_gp_reg_mapping = { 0/*avoid warning "maybe used uninitialized" */ };
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
  if (NULL == buf) {
    LIBXSMM_HANDLE_ERROR(io_code, LIBXSMM_ERR_BUFFER_TOO_SMALL);
    return;
  }

  if ( io_code->code_type > 1 )
  {
     unsigned int i = io_code->code_size;
     unsigned int m = i_packed_trsm_desc->m;
     unsigned int n = i_packed_trsm_desc->n;
     unsigned int lda = i_packed_trsm_desc->lda;
     unsigned int ldb = i_packed_trsm_desc->ldb;
     char trans = i_packed_trsm_desc->transa;
     char side = i_packed_trsm_desc->side;
     char uplo = i_packed_trsm_desc->uplo;
     char diag = i_packed_trsm_desc->diag;
     const unsigned int layout = (unsigned int)i_packed_trsm_desc->layout;
     unsigned int datasz = (unsigned int)i_packed_trsm_desc->typesize;
     const double alpha = (8 == datasz ? i_packed_trsm_desc->alpha.d : ((double)i_packed_trsm_desc->alpha.s));
     unsigned int m1=m, n1=n;
     unsigned int j, k;
     /*int REGSIZE;*/
     int numb = 0;
     int scalealpha = 0;
     int nounit=0;
     char regset = 'y';

     if ( layout == 101 )
     {
        if (i_packed_trsm_desc->side == 'L' || i_packed_trsm_desc->side == 'l' ) side = 'R';
        else side = 'L';
        if (i_packed_trsm_desc->uplo == 'L' || i_packed_trsm_desc->uplo == 'l' ) uplo = 'U';
        else uplo = 'L';
        m1 = n; n1 = m;
     }
#ifdef GENERATOR_PACKED_TRSM_DEBUG
printf("Inside libxsmm_generator_packed_trsm_avx_avx512_kernel: %c%c%c%c m=%d n=%d lay=102 alpha=%g datasz=%d\n",side,uplo,trans,diag,m1,n1,alpha,datasz);
#endif
     if ( ( datasz !=4 ) && (datasz != 8) )
     {
        fprintf(stderr,"Expecting a datasize of 4 or 8 but got %u\n",datasz);
        exit(-1);
     }
     if ( avx512 < 0 )
     {
        fprintf(stderr,"Expecting a nonnegative number for avx512: %i\n",avx512);
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
     } else if ( datasz == 8 && avx512 > 0 )
     {
        numb = 8;
        regset = 'z';
     }

     if ( LIBXSMM_FEQ(0, alpha) )
     {
        compact_set_zero_ ( io_code, 0, numb, datasz, regset );
        for ( j = 1; j <= n1; j++ )
        {
           for ( i = 1; i <= m1; i++ )
           {
              compact_store_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
           }
        }
        i = io_code->code_size;
        buf[i++] = 0xc3; /* retq */
        io_code->code_size = i;
        return;
     }
     if ( LIBXSMM_NEQ(1, alpha) )
     {
        compact_load_parameter_ ( io_code, alpha, 2, numb, regset );
     }
     nounit = ( (diag=='N') || (diag=='n') );

     if ( (side=='L') || (side=='l') )
     {
        if ( (trans=='N') || (trans=='n') )
        {
           if ( (uplo=='U') || (uplo=='u') )
           {
              /* Do LUN* cases: B<- alpha*inv(A)*B */
              if ( nounit )
              {
                  compact_set_one_ ( io_code, 15, numb, datasz, regset );
                  for ( i = 1; i <= m1; i++ )
                  {
                     compact_load_matrix1_ ( io_code, lda, i, i, 3, numb, datasz, regset );
                     compact_divide_two_nums_ ( io_code, 15, 3, 3 , numb, regset );
                     compact_store_matrix3_ ( io_code, m1, i, 1, 3, numb, datasz, regset );
                  }
              }
              for ( j = 1; j <= n1; j++ )
              {
                 if ( LIBXSMM_NEQ(1, alpha) )
                 {
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
                    }
                 }
                 for ( k = m1; k >= 1; k-- )
                 {
                    compact_load_matrix2_ ( io_code, ldb, k, j, 0, numb, datasz, regset );
                    if ( nounit )
                    {
#if 0
                       compact_load_matrix1_ ( io_code, lda, k, k, 1, numb, datasz, regset );
                       compact_divide_two_nums_ ( io_code, 0, 1, 0, numb, regset );
#else
                       compact_load_matrix3_ ( io_code, m1, k, 1, 1, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 0, 1, 0, numb, regset );
#endif
                       compact_store_matrix2_ ( io_code, ldb, k, j, 0, numb, datasz, regset );
                    }
                    for ( i = 1; i <= k-1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, j, 1, numb, datasz, regset );
                       compact_load_matrix1_ ( io_code, lda, i, k, 3, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 1, 0, 3, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, j, 1, numb, datasz, regset );
                    }
                 }
              }
           } else {
              /* Do LLN* cases: B <- alpha * inv(A)*B */
#if 0
#define USE_XCT_LLNN
#endif
#ifdef USE_XCT_LLNN
              int done = 0;
              if ( (avx512==0) && (alpha==1.0) )
              {
#include "generator_compact_xct_avx2_lln.h"
              done = 1;
              }
              if ( done == 0 )
#endif
              {

              /* Do LLN* cases: B <- alpha * inv(A)*B */
              if ( nounit )
              {
                  compact_set_one_ ( io_code, 15, numb, datasz, regset );
                  for ( i = 1; i <= m1; i++ )
                  {
                     compact_load_matrix1_ ( io_code, lda, i, i, 3, numb, datasz, regset );
                     compact_divide_two_nums_ ( io_code, 15, 3, 3 , numb, regset );
                     compact_store_matrix3_ ( io_code, m1, i, 1, 3, numb, datasz, regset );
                  }
              }

              for ( j = 1; j <= n1; j+=3 )
              {
                 for ( k = 1; k <= m1; k+=2 )
                 {
                    scalealpha = 0;
                    if ( LIBXSMM_NEQ(1, alpha) && (k==1) ) scalealpha = 1;
                    compact_load_matrix2_ ( io_code, ldb, k, j, 0, numb, datasz, regset );
                    if ( j+1 <= n1 ) compact_load_matrix2_ ( io_code, ldb, k, j+1, 4, numb, datasz, regset );
                    if ( j+2 <= n1 ) compact_load_matrix2_ ( io_code, ldb, k, j+2, 7, numb, datasz, regset );
                    if ( scalealpha == 1 ) {
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
                       if ( j+1 <= n1 ) compact_mult_two_nums_ ( io_code, 4, 2, 4, numb, regset );
                       if ( j+2 <= n1 ) compact_mult_two_nums_ ( io_code, 7, 2, 7, numb, regset );
                    }
                    if ( nounit ) {
                       compact_load_matrix3_ ( io_code, m1, k, 1, 1, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 0, 1, 0, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, k, j, 0, numb, datasz, regset );
                       if ( j+1 <= n1 ) {
                          compact_mult_two_nums_ ( io_code, 4, 1, 4, numb, regset );
                          compact_store_matrix2_ ( io_code, ldb, k, j+1, 4, numb, datasz, regset );
                       }
                       if ( j+2 <= n1 ) {
                          compact_mult_two_nums_ ( io_code, 7, 1, 7, numb, regset );
                          compact_store_matrix2_ ( io_code, ldb, k, j+2, 7, numb, datasz, regset );
                       }
                    }
                    if ( k+1 <= m1 ) {
                       compact_load_matrix2_ ( io_code, ldb, k+1, j, 10, numb, datasz, regset );
                       if ( scalealpha == 1 ) {
                          compact_mult_two_nums_ ( io_code, 10, 2, 10, numb, regset );
                       }
                       compact_load_matrix1_ ( io_code, ldb, k+1, k, 3, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 10, 0, 3, numb, regset );
                       if ( j+1 <= n1 ) {
                          compact_load_matrix2_ ( io_code, ldb, k+1, j+1, 14, numb, datasz, regset );
                          if ( scalealpha == 1 ) {
                             compact_mult_two_nums_ ( io_code, 14, 2, 14, numb, regset );
                          }
                          compact_load_matrix1_ ( io_code, ldb, k+1, k, 3, numb, datasz, regset );
                          compact_fms_cminusab_ ( io_code, 14, 4, 3, numb, regset );
                       }
                       if ( j+2 <= n1 ) {
                          compact_load_matrix2_ ( io_code, ldb, k+1, j+2, 9, numb, datasz, regset );
                          if ( scalealpha == 1 ) {
                             compact_mult_two_nums_ ( io_code, 9, 2, 9, numb, regset );
                          }
                          compact_fms_cminusab_ ( io_code, 9, 7, 3, numb, regset );
                       }
                       if ( nounit ) {
                          compact_load_matrix3_ ( io_code, m1, k+1, 1, 11, numb, datasz, regset );
                          compact_mult_two_nums_ ( io_code, 10, 11, 10, numb, regset );
                          compact_store_matrix2_ ( io_code, ldb, k+1, j, 10, numb, datasz, regset );
                          if ( j+1 <= n1 ) {
                             compact_mult_two_nums_ ( io_code, 14, 11, 14, numb, regset );
                             compact_store_matrix2_ ( io_code, ldb, k+1, j+1, 14, numb, datasz, regset );
                          }
                          if ( j+2 <= n1 ) {
                             compact_mult_two_nums_ ( io_code, 9, 11, 9, numb, regset );
                             compact_store_matrix2_ ( io_code, ldb, k+1, j+2, 9, numb, datasz, regset );
                          }
                       }
                    }
                    for ( i = k+2; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, j, 1, numb, datasz, regset );
                       if ( scalealpha == 1 ) {
                          compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
                       }
                       compact_load_matrix1_ ( io_code, ldb, i, k, 3, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 1, 0, 3, numb, regset );
                       if ( k+1 > m1 ) {
                          compact_store_matrix2_ ( io_code, ldb, i, j, 1, numb, datasz, regset );
                       }
                       if ( j+1 <= n1 ) {
                          compact_load_matrix2_ ( io_code, ldb, i, j+1, 6, numb, datasz, regset );
                          if ( scalealpha == 1 ) {
                             compact_mult_two_nums_ ( io_code, 6, 2, 6, numb, regset );
                          }
                          compact_fms_cminusab_ ( io_code, 6, 4, 3, numb, regset );
                          if ( k+1 > m1 ) {
                              compact_store_matrix2_ ( io_code, ldb, i, j+1, 6, numb, datasz, regset );
                          }
                       }
                       if ( j+2 <= n1 ) {
                          compact_load_matrix2_ ( io_code, ldb, i, j+2, 12, numb, datasz, regset );
                          if ( scalealpha == 1 ) {
                             compact_mult_two_nums_ ( io_code, 12, 2, 12, numb, regset );
                          }
                          compact_fms_cminusab_ ( io_code, 12, 7, 3, numb, regset );
                          if ( k+1 > m1 ) {
                              compact_store_matrix2_ ( io_code, ldb, i, j+2, 12, numb, datasz, regset );
                          }
                       }
                       if ( k+1 <= m1 ) {
                          compact_load_matrix1_ ( io_code, ldb, i, k+1, 13, numb, datasz, regset );
                          compact_fms_cminusab_ ( io_code, 1, 10, 13, numb, regset );
                          compact_store_matrix2_ ( io_code, ldb, i, j, 1, numb, datasz, regset );
                          if ( j+1 <= n1 ) {
                             compact_fms_cminusab_ ( io_code, 6, 14, 13, numb, regset );
                             compact_store_matrix2_ ( io_code, ldb, i, j+1, 6, numb, datasz, regset );
                          }
                          if ( j+2 <= n1 ) {
                             compact_fms_cminusab_ ( io_code, 12, 9, 13, numb, regset );
                             compact_store_matrix2_ ( io_code, ldb, i, j+2, 12, numb, datasz, regset );
                          }
                       }
                    } /* for i LLN main loop */
                 }    /* for k LLN loop */
              }       /* for j LLN loop */
              }       /* Call XCT LLN kernel or not */
           } /* uplo */
        } else {
           if ( (uplo=='U') || (uplo=='u') )
           {
              /* Do LUT* cases: B<- alpha*inv(A^T)*B */
#define LUT_RECIPROCATE
#ifdef LUT_RECIPROCATE
              if ( nounit )
              {
                  compact_set_one_ ( io_code, 15, numb, datasz, regset );
                  for ( i = 1; i <= m1; i++ )
                  {
                     compact_load_matrix1_ ( io_code, lda, i, i, 3, numb, datasz, regset );
                     compact_divide_two_nums_ ( io_code, 15, 3, 3 , numb, regset );
                     compact_store_matrix3_ ( io_code, m1, i, 1, 3, numb, datasz, regset );
                  }
              }
#endif
#define LUT_N2
#ifdef LUT_N2
              for ( j = 1; j <= n1; j+=2 )
#else
              for ( j = 1; j <= n1; j++  )
#endif
              {
                 for ( i = 1; i <= m1; i+=2 )
                 {
                    compact_load_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
#ifdef LUT_N2
                    if ( j+1 <= n1 ) compact_load_matrix2_ ( io_code, ldb, i, j+1, 4, numb, datasz, regset );
#endif
                    if ( i+1 <= m1 ) compact_load_matrix2_ ( io_code, ldb, i+1, j, 7, numb, datasz, regset );
#ifdef LUT_N2
                    if ((i+1<=m1)&&(j+1<=n1)) compact_load_matrix2_ ( io_code, ldb, i+1, j+1, 9, numb, datasz, regset );
#endif
                    if ( LIBXSMM_NEQ(1, alpha) ) {
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
#ifdef LUT_N2
                       if ( j+1 <= n1 ) compact_mult_two_nums_ ( io_code, 4, 2, 4, numb, regset );
#endif
                       if ( i+1 <= m1 ) compact_mult_two_nums_ ( io_code, 7, 2, 7, numb, regset );
#ifdef LUT_N2
                       if ((i+1<=m1)&&(j+1<=n1)) compact_mult_two_nums_ ( io_code, 9, 2, 9, numb, regset );
#endif
                    }
                    for ( k = 1; k <= i-1; k++ ) {
                       compact_load_matrix2_( io_code, ldb, k, j, 1, numb, datasz, regset );
                       compact_load_matrix1_( io_code, lda, k, i, 3, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 0, 3, 1, numb, regset );
#ifdef LUT_N2
                       if ( j+1 <= n1 ) {
                           compact_load_matrix2_( io_code, ldb, k, j+1, 5, numb, datasz, regset );
                           compact_fms_cminusab_ ( io_code, 4, 3, 5, numb, regset );
                       }
#endif
                       if ( i+1 <= m1 ) {
                           compact_load_matrix1_( io_code, lda, k, i+1, 8, numb, datasz, regset );
                           compact_fms_cminusab_ ( io_code, 7, 8, 1, numb, regset );
                       }
#ifdef LUT_N2
                       if ((i+1<=m1)&&(j+1<=n1)) {
                           compact_fms_cminusab_ ( io_code, 9, 8, 5, numb, regset );
                       }
#endif
                    }
                    if ( nounit )
                    {
#ifdef LUT_RECIPROCATE
                       compact_load_matrix3_ ( io_code, m1, i, 1, 3, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 0, 3, 0, numb, regset );
  #ifdef LUT_N2
                       if ( j+1 <= n1 ) compact_mult_two_nums_ ( io_code, 4, 3, 4, numb, regset );
  #endif
#else
                       compact_load_matrix1_ ( io_code, lda, i, i, 3, numb, datasz, regset );
                       compact_divide_two_nums_ ( io_code, 0, 3, 0, numb, regset );
  #ifdef LUT_N2
                       if ( j+1 <= n1 ) compact_divide_two_nums_ ( io_code, 4, 3, 4, numb, regset );
  #endif
#endif
                    }
                    compact_store_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
#ifdef LUT_N2
                    if ( j+1 <= n1 ) compact_store_matrix2_ ( io_code, ldb, i, j+1, 4, numb, datasz, regset );
#endif
                    if ( i+1 <= m1 ) {
                       compact_load_matrix1_( io_code, lda, i, i+1, 8, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 7, 8, 0, numb, regset );
#ifdef LUT_N2
                       if ( j+1 <= n1 ) compact_fms_cminusab_ ( io_code, 9, 8, 4, numb, regset );
#endif
                       if ( nounit ) {
#ifdef LUT_RECIPROCATE
                          compact_load_matrix3_ ( io_code, m1, i+1, 1, 3, numb, datasz, regset );
                          compact_mult_two_nums_ ( io_code, 7, 3, 7, numb, regset );
  #ifdef LUT_N2
                          if ( j+1 <= n1 ) compact_mult_two_nums_ ( io_code, 9, 3, 9, numb, regset );
  #endif
#else
                          compact_load_matrix1_ ( io_code, lda, i+1, i+1, 3, numb, datasz, regset );
                          compact_divide_two_nums_ ( io_code, 7, 3, 7, numb, regset );
  #ifdef LUT_N2
                          if ( j+1 <= n1 ) compact_divide_two_nums_ ( io_code, 9, 3, 9, numb, regset );
  #endif
#endif
                       }
                       compact_store_matrix2_ ( io_code, ldb, i+1, j, 7, numb, datasz, regset );
#ifdef LUT_N2
                       if ( j+1 <= n1 ) compact_store_matrix2_ ( io_code, ldb, i+1, j+1, 9, numb, datasz, regset );
#endif
                    }
                 }
              }
           } else {
              /* Do LLT* cases: B <- alpha * inv(A)*B */
#define LLT_N2
#define LLT_M2
#if 1
#define LLT_RECIPROCATE
#endif

#ifdef LLT_RECIPROCATE
              if ( nounit )
              {
                  compact_set_one_ ( io_code, 15, numb, datasz, regset );
                  for ( i = 1; i <= m1; i++ )
                  {
                     compact_load_matrix1_ ( io_code, lda, i, i, 3, numb, datasz, regset );
                     compact_divide_two_nums_ ( io_code, 15, 3, 3 , numb, regset );
                     compact_store_matrix3_ ( io_code, m1, i, 1, 3, numb, datasz, regset );
                  }
              }
#endif
#ifdef LLT_N2
              for ( j = 1; j <= n1; j+=2 )
#else
              for ( j = 1; j <= n1; j+=1 )
#endif
              {
#ifdef LLT_M2
                 for ( i = m1; i >= 1; i-=2 )
#else
                 for ( i = m1; i >= 1; i-=1 )
#endif
                 {
                    compact_load_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
#ifdef LLT_M2
                    if ( i-1 >= 1 ) compact_load_matrix2_ ( io_code, ldb, i-1, j, 4, numb, datasz, regset );
#endif
#ifdef LLT_N2
                    if ( j+1 <= n1 ) compact_load_matrix2_ ( io_code, ldb, i, j+1, 7, numb, datasz, regset );
#endif
#if defined(LLT_N2) && defined(LLT_M2)
                    if ( (i-1>=1) && (j+1<=n1) ) compact_load_matrix2_ ( io_code, ldb, i-1, j+1, 10, numb, datasz, regset );
#endif
                    if ( LIBXSMM_NEQ(1, alpha) )
                    {
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
#ifdef LLT_M2
                       if ( i-1 >= 1 ) compact_mult_two_nums_ ( io_code, 4, 2, 4, numb, regset );
#endif
#ifdef LLT_N2
                       if ( j+1 <= n1) compact_mult_two_nums_ ( io_code, 7, 2, 7, numb, regset );
#endif
#if defined(LLT_N2) && defined(LLT_M2)
                       if ((i-1>=1)&&(j+1<=n1)) compact_mult_two_nums_ ( io_code, 10, 2, 10, numb, regset );
#endif
                    }
                    for ( k = i+1; k <= m1; k++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, k, j, 1, numb, datasz, regset );
                       compact_load_matrix1_ ( io_code, lda, k, i, 3, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 0, 3, 1, numb, regset );
#ifdef LLT_M2
                       if ( i-1 >= 1 ) {
                          compact_load_matrix1_ ( io_code, lda, k, i-1, 6, numb, datasz, regset );
                          compact_fms_cminusab_ ( io_code, 4, 6, 1, numb, regset );
                       }
#endif
#ifdef LLT_N2
                       if ( j+1 <= n1) {
                          compact_load_matrix2_ ( io_code, ldb, k, j+1, 8, numb, datasz, regset );
  #if 0
                          compact_load_matrix1_ ( io_code, lda, k, i, 9, numb, datasz, regset );
                          compact_fms_cminusab_ ( io_code, 7, 9, 8, numb, regset );
  #else
                          compact_fms_cminusab_ ( io_code, 7, 3, 8, numb, regset );
  #endif
                       }
#endif
#if defined(LLT_N2) && defined(LLT_M2)
                       if ((i-1>=1)&&(j+1<=n1)) {
  #if 0
                          compact_load_matrix2_ ( io_code, ldb, k, j+1, 11, numb, datasz, regset );
                          compact_load_matrix1_ ( io_code, lda, k, i-1, 12, numb, datasz, regset );
                          compact_fms_cminusab_ ( io_code, 10, 12, 11, numb, regset );
  #else
                          compact_fms_cminusab_ ( io_code, 10, 6, 8 , numb, regset );
  #endif

                       }
#endif
                    }
                    if ( nounit )
                    {
#ifndef LLT_RECIPROCATE
                       compact_load_matrix1_ ( io_code, lda, i, i, 3, numb, datasz, regset );
                       compact_divide_two_nums_ ( io_code, 0, 3, 0, numb, regset );
   #ifdef LLT_N2
                       if ( j+1 <= n1 ) {
                          compact_divide_two_nums_ ( io_code, 7, 3, 7, numb, regset );
                       }
   #endif
#else
                       compact_load_matrix3_ ( io_code, m1, i, 1, 3, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 0, 3, 0, numb, regset );
   #ifdef LLT_N2
                       if ( j+1 <= n1 ) {
                          compact_mult_two_nums_ ( io_code, 7, 3, 7, numb, regset );
                       }
   #endif
#endif
                    }
                    compact_store_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
#ifdef LLT_M2
                    if ( i-1 >= 1 ) {
                       compact_load_matrix1_ ( io_code, lda, i, i-1, 6, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 4, 6, 0, numb, regset );
                       if ( nounit ) {
   #ifndef LLT_RECIPROCATE
                          compact_load_matrix1_ ( io_code, lda, i-1, i-1, 6, numb, datasz, regset );
                          compact_divide_two_nums_ ( io_code, 4, 6, 4, numb, regset );
   #else
                          compact_load_matrix3_ ( io_code, m1, i-1, 1, 6, numb, datasz, regset );
                          compact_mult_two_nums_ ( io_code, 4, 6, 4, numb, regset );
   #endif
                       }
                       compact_store_matrix2_ ( io_code, ldb, i-1, j, 4, numb, datasz, regset );
                    }
#endif
#ifdef LLT_N2
                    if ( j+1 <= n1) compact_store_matrix2_ ( io_code, ldb, i, j+1, 7, numb, datasz, regset );
#endif
#if defined(LLT_N2) && defined(LLT_M2)
                    if ((i-1>=1)&&(j+1<=n1)) {
                       compact_load_matrix1_ ( io_code, lda, i, i-1, 12, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 10, 12, 7, numb, regset );
                       if ( nounit ) {
   #ifdef LLT_RECIPROCATE
                          compact_mult_two_nums_ ( io_code, 10, 6, 10, numb, regset );
   #else
                          compact_load_matrix1_ ( io_code, lda, i-1, i-1, 12, numb, datasz, regset );
                          compact_divide_two_nums_ ( io_code, 10, 12, 10, numb, regset );
   #endif
                       }
                       compact_store_matrix2_ ( io_code, ldb, i-1, j+1, 10, numb, datasz, regset );
                    }
#endif
                 }
              }
            } /* uplo */
         } /* trans */
     } else {
        compact_set_one_ ( io_code, 5, numb, datasz, regset );
        if ( (trans=='N') || (trans=='n') )
        {
           if ( (uplo=='U') || (uplo=='u') )
           {
              /* Do RUN* cases: B<- alpha*B*inv(A) */
              if ( nounit )
              {
                  compact_set_one_ ( io_code, 15, numb, datasz, regset );
                  for ( i = 1; i <= n1; i++ )
                  {
                     compact_load_matrix1_ ( io_code, lda, i, i, 3, numb, datasz, regset );
                     compact_divide_two_nums_ ( io_code, 15, 3, 3 , numb, regset );
                     compact_store_matrix3_ ( io_code, n1, i, 1, 3, numb, datasz, regset );
                  }
              }
              for ( j = 1; j <= n1; j+=2 )
              {
                 if ( LIBXSMM_NEQ(1, alpha) && (j==1) )
                 {
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
                       if ( j+1 <= n1 ) compact_load_matrix2_ ( io_code, ldb, i, j+1, 1, numb, datasz, regset );
#if 0
                       if ( j+2 <= n1 ) compact_load_matrix2_ ( io_code, ldb, i, j+2, 3, numb, datasz, regset );
                       if ( j+3 <= n1 ) compact_load_matrix2_ ( io_code, ldb, i, j+3, 4, numb, datasz, regset );
#endif
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
                       if ( j+1 <= n1 ) compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
#if 0
                       if ( j+2 <= n1 ) compact_mult_two_nums_ ( io_code, 3, 2, 3, numb, regset );
                       if ( j+3 <= n1 ) compact_mult_two_nums_ ( io_code, 4, 2, 4, numb, regset );
#endif
                       compact_store_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
                       if ( j+1 <= n1 ) compact_store_matrix2_ ( io_code, ldb, i, j+1, 1, numb, datasz, regset );
#if 0
                       if ( j+2 <= n1 ) compact_store_matrix2_ ( io_code, ldb, i, j+2, 3, numb, datasz, regset );
                       if ( j+3 <= n1 ) compact_store_matrix2_ ( io_code, ldb, i, j+3, 4, numb, datasz, regset );
#endif
                    }
                 }
                 for ( k = 1; k <= j-1; k++ )
                 {
                    if ( (k==j-1) && (nounit) ) {
                       compact_load_matrix3_ ( io_code, n1, j, 1, 5, numb, datasz, regset );
                    }
                    compact_load_matrix1_ ( io_code, lda, k, j, 3, numb, datasz, regset );
                    if ( j+1 <= n1 ) compact_load_matrix1_ ( io_code, lda, k, j+1, 6, numb, datasz, regset );
#if 0
                    if ( j+2 <= n1 ) compact_load_matrix1_ ( io_code, lda, k, j+2, 10, numb, datasz, regset );
                    if ( j+3 <= n1 ) compact_load_matrix1_ ( io_code, lda, k, j+3, 12, numb, datasz, regset );
#endif
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, j, 1, numb, datasz, regset );
                       if (j+1<=n1) compact_load_matrix2_ ( io_code, ldb, i, j+1, 7, numb, datasz, regset );
#if 0
                       if (j+2<=n1) compact_load_matrix2_ ( io_code, ldb, i, j+2, 11, numb, datasz, regset );
                       if (j+3<=n1) compact_load_matrix2_ ( io_code, ldb, i, j+3, 13, numb, datasz, regset );
#endif
                       if ((k==1)&&LIBXSMM_NEQ(1,alpha)) compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
                       if ((j+1<=n1)&&(k==1)&&LIBXSMM_NEQ(1,alpha)) compact_mult_two_nums_ ( io_code, 7, 2, 7, numb, regset );
#if 0
                       if ((j+2<=n1)&&(k==1)&&LIBXSMM_NEQ(1,alpha)) compact_mult_two_nums_ ( io_code, 11, 2, 11, numb, regset );
                       if ((j+3<=n1)&&(k==1)&&LIBXSMM_NEQ(1,alpha)) compact_mult_two_nums_ ( io_code, 13, 2, 13, numb, regset );
#endif
                       compact_load_matrix2_ ( io_code, ldb, i, k, 4, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 1, 3, 4, numb, regset );
                       if (j+1<=n1) compact_fms_cminusab_ ( io_code, 7, 6, 4, numb, regset );
#if 0
                       if (j+2<=n1) compact_fms_cminusab_ ( io_code, 11, 10, 4, numb, regset );
                       if (j+3<=n1) compact_fms_cminusab_ ( io_code, 13, 12, 4, numb, regset );
#endif
                       if ( (k==j-1) && (nounit) ) compact_mult_two_nums_ ( io_code, 1, 5, 1, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, j, 1, numb, datasz, regset );
                       if (j+1<=n1) compact_store_matrix2_ ( io_code, ldb, i, j+1, 7, numb, datasz, regset );
#if 0
                       if (j+2<=n1) compact_store_matrix2_ ( io_code, ldb, i, j+2, 11, numb, datasz, regset );
                       if (j+3<=n1) compact_store_matrix2_ ( io_code, ldb, i, j+3, 13, numb, datasz, regset );
#endif
                    }
                 }
                 if ( j+1 <= n1 ) {
                    for ( k = j; k <= j; k++ ) {
                       compact_load_matrix1_ ( io_code, lda, k, j+1, 6, numb, datasz, regset );
                       if ( j==1 && nounit ) compact_load_matrix3_ ( io_code, n1, j, 1, 9, numb, datasz, regset );
                       if ( nounit ) compact_load_matrix3_ ( io_code, n1, j+1, 1, 1, numb, datasz, regset );
                       for ( i = 1; i <= m1; i++ ) {
                          compact_load_matrix2_ ( io_code, ldb, i, j+1, 7, numb, datasz, regset );
                          compact_load_matrix2_ ( io_code, ldb, i, k, 8, numb, datasz, regset );
                          if (j==1 && nounit) {
                             compact_mult_two_nums_ ( io_code, 8, 9, 8, numb, regset );
                             compact_store_matrix2_ ( io_code, ldb, i, k, 8, numb, datasz, regset );
                          }

                          compact_fms_cminusab_ ( io_code, 7, 6, 8, numb, regset );
                          if (nounit) compact_mult_two_nums_ ( io_code, 7, 1, 7, numb, regset );
                          compact_store_matrix2_ ( io_code, ldb, i, j+1, 7, numb, datasz, regset );
                       }
                    }
                 }
#if 0
                 if ( j+2 <= n1 ) {
                    for ( k = j; k <= j+1; k++ ) {
                       compact_load_matrix1_ ( io_code, lda, k, j+2, 6, numb, datasz, regset );
                       if ( (k==j+1) && nounit ) compact_load_matrix3_ ( io_code, n1, j+2, 1, 1, numb, datasz, regset );
                       for ( i = 1; i <= m1; i++ ) {
                          compact_load_matrix2_ ( io_code, ldb, i, j+2, 7, numb, datasz, regset );
                          compact_load_matrix2_ ( io_code, ldb, i, k, 8, numb, datasz, regset );
                          compact_fms_cminusab_ ( io_code, 7, 6, 8, numb, regset );
                          if ((k==j+1)&& nounit) compact_mult_two_nums_ ( io_code, 7, 1, 7, numb, regset );
                          compact_store_matrix2_ ( io_code, ldb, i, j+2, 7, numb, datasz, regset );
                       }
                    }
                 }
                 if ( j+3 <= n1 ) {
                    for ( k = j; k <= j+2; k++ ) {
                       compact_load_matrix1_ ( io_code, lda, k, j+3, 6, numb, datasz, regset );
                       if ( (k==j+2) && nounit ) compact_load_matrix3_ ( io_code, n1, j+3, 1, 1, numb, datasz, regset );
                       for ( i = 1; i <= m1; i++ ) {
                          compact_load_matrix2_ ( io_code, ldb, i, j+3, 7, numb, datasz, regset );
                          compact_load_matrix2_ ( io_code, ldb, i, k, 8, numb, datasz, regset );
                          compact_fms_cminusab_ ( io_code, 7, 6, 8, numb, regset );
                          if ((k==j+2)&& nounit) compact_mult_two_nums_ ( io_code, 7, 1, 7, numb, regset );
                          compact_store_matrix2_ ( io_code, ldb, i, j+3, 7, numb, datasz, regset );
                       }
                    }
                 }
#endif
              }
           } else {
              /* Do RLN* cases: B <- alpha * B * inv(A) */
              for ( j = n1; j >= 1; j-- )
              {
                 if ( LIBXSMM_NEQ(1, alpha) )
                 {
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
                    }
                 }
                 for ( k = j+1; k <= n1; k++ )
                 {
                    compact_load_matrix1_ ( io_code, lda, k, j, 3, numb, datasz, regset );
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, j, 1, numb, datasz, regset );
                       compact_load_matrix2_ ( io_code, ldb, i, k, 4, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 1, 3, 4, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, j, 1, numb, datasz, regset );
                    }
                 }
                 if ( nounit )
                 {
                    compact_load_matrix1_( io_code, lda, j, j, 1, numb, datasz, regset );
                    compact_divide_two_nums_ ( io_code, 5, 1, 1, numb, regset );
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, j, 3, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 1, 3, 3, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, j, 3, numb, datasz, regset );
                    }
                 }
              }
           } /* uplo */
        } else {
           if ( (uplo=='U') || (uplo=='u') )
           {
              /* Do RUT* cases: B<- alpha*B *inv(A^T) */
              for ( k = n1; k >= 1; k-- )
              {
                 if ( nounit )
                 {
                    compact_load_matrix1_( io_code, lda, k, k, 1, numb, datasz, regset );
                    compact_divide_two_nums_ ( io_code, 5, 1, 1, numb, regset );
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, k, 3, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 1, 3, 3, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, k, 3, numb, datasz, regset );
                    }
                 }
                 for ( j = 1; j <= k-1; j++ )
                 {
                    compact_load_matrix1_ ( io_code, lda, j, k, 1, numb, datasz, regset );
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
                       compact_load_matrix2_ ( io_code, ldb, i, k, 3, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 0, 1, 3, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
                    }
                 }
                 if ( LIBXSMM_NEQ(1, alpha) )
                 {
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, k, 0, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, k, 0, numb, datasz, regset );
                    }
                 }
              }
           } else {
              /* Do RLT* cases: B <- alpha * B *inv(A^T) */
              for ( k = 1; k <= n1; k++ )
              {
                 if ( nounit )
                 {
                    compact_load_matrix1_ ( io_code, lda, k, k, 1, numb, datasz, regset );
                    compact_divide_two_nums_ ( io_code, 5, 1, 1, numb, regset );
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, k, 3, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 1, 3, 3, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, k, 3, numb, datasz, regset );
                    }
                 }
                 for ( j = k+1; j <= n1; j++ )
                 {
                    compact_load_matrix1_ ( io_code, lda, j, k, 1, numb, datasz, regset );
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_( io_code, ldb, i, j, 0, numb, datasz, regset );
                       compact_load_matrix2_( io_code, ldb, i, k, 3, numb, datasz, regset );
                       compact_fms_cminusab_ ( io_code, 0, 1, 3, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, j, 0, numb, datasz, regset );
                    }
                 }
                 if ( LIBXSMM_NEQ(1, alpha) )
                 {
                    for ( i = 1; i <= m1; i++ )
                    {
                       compact_load_matrix2_ ( io_code, ldb, i, k, 0, numb, datasz, regset );
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
                       compact_store_matrix2_ ( io_code, ldb, i, k, 0, numb, datasz, regset );
                    }
                 }
              }
           } /* uplo */
        } /* trans */
     } /* side */
  }

  { int i = io_code->code_size;
    buf[i++] = 0xc3; /* retq */
    io_code->code_size = i;
  }
  /*  close asm: note that we really didn't need to push everything */
/*
  libxsmm_x86_instruction_close_stream_transpose( io_code, i_arch );
*/
#ifdef GENERATOR_PACKED_TRSM_DEBUG
  printf("done with m=%d n=%d i=%d\n",i_trans_desc->m,i_trans_desc->n,io_code->code_size);
#endif

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

