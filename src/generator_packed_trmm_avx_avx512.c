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
#include "generator_packed_trmm_avx_avx512.h"
#include "generator_x86_instructions.h"
#include "generator_packed_aux.h"
#include "generator_common.h"
#include "libxsmm_main.h"

#if 0
# define GENERATOR_PACKED_TRMM_DEBUG
#endif


LIBXSMM_API_INTERN
void libxsmm_generator_packed_trmm_avx_avx512_kernel( libxsmm_generated_code*        io_code,
                                                       const libxsmm_trmm_descriptor* i_packed_trmm_desc,
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
  libxsmm_trmm_gp_reg_mapping l_gp_reg_mapping = { 0/*avoid warning "maybe used uninitialized" */ };
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
     unsigned int m = i_packed_trmm_desc->m;
     unsigned int n = i_packed_trmm_desc->n;
     unsigned int lda = i_packed_trmm_desc->lda;
     unsigned int ldb = i_packed_trmm_desc->ldb;
     char trans = i_packed_trmm_desc->transa;
     char side = i_packed_trmm_desc->side;
     char uplo = i_packed_trmm_desc->uplo;
     char diag = i_packed_trmm_desc->diag;
#if defined(_WIN32) || defined(__CYGWIN__)
     unsigned char areg = LIBXSMM_X86_GP_REG_RCX;
     unsigned char breg = LIBXSMM_X86_GP_REG_RDX;
#else
     unsigned char areg = LIBXSMM_X86_GP_REG_RDI;
     unsigned char breg = LIBXSMM_X86_GP_REG_RSI;
#endif
     const unsigned int lay = (unsigned int)i_packed_trmm_desc->layout;
     unsigned int datasz = (unsigned int)i_packed_trmm_desc->typesize;
     const double alpha = (8 == datasz ? i_packed_trmm_desc->alpha.d : ((double)i_packed_trmm_desc->alpha.s));
     unsigned int m1=m, n1=n;
     unsigned int j, k;
     /*int REGSIZE;*/
     int numb = 0;
     /*int scalealpha = 0;*/
     int nounit=0;
     char regset = 'y';

     if ( lay == 101 )
     {
        if (i_packed_trmm_desc->side == 'L' || i_packed_trmm_desc->side == 'l' ) side = 'R';
        else side = 'L';
        if (i_packed_trmm_desc->uplo == 'L' || i_packed_trmm_desc->uplo == 'l' ) uplo = 'U';
        else uplo = 'L';
        m1 = n; n1 = m;
     }
#ifdef GENERATOR_PACKED_TRMM_DEBUG
printf("Inside libxsmm_generator_packed_trmm_avx_avx512_kernel: %c%c%c%c m=%d n=%d lay=%d alpha=%g datasz=%d\n",side,uplo,trans,diag,m1,n1,lay,alpha,datasz);
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
              compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
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
              for ( j = 1; j <= n1; j+=3 )
              {
                 for ( k = 1; k <= m1; k++ )
                 {
                    compact_load_matrix_gen_ ( io_code, 0, ldb, k, j, 0, numb, datasz, regset, breg );
                    if ( j+1 <= n1 ) compact_load_matrix_gen_ ( io_code, 0, ldb, k, j+1, 4, numb, datasz, regset, breg );
                    if ( j+2 <= n1 ) compact_load_matrix_gen_ ( io_code, 0, ldb, k, j+2, 7, numb, datasz, regset, breg );
                    if ( LIBXSMM_NEQ(1, alpha) ) {
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
                       if ( j+1 <= n1 ) compact_mult_two_nums_ ( io_code, 4, 2, 4, numb, regset );
                       if ( j+2 <= n1 ) compact_mult_two_nums_ ( io_code, 7, 2, 7, numb, regset );
                    }
                    for ( i = 1; i <= k-1; i++ )
                    {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, j, 3, numb, datasz, regset, breg );
                       compact_load_matrix_gen_ ( io_code, 0, lda, i, k, 1, numb, datasz, regset, areg );
                       compact_fma_cplusab_ ( io_code, 3, 0, 1, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 3, numb, datasz, regset, breg );
                       if ( j+1 <= n1 ) {
                          compact_load_matrix_gen_ ( io_code, 0, ldb, i, j+1, 6, numb, datasz, regset, breg );
                          compact_fma_cplusab_ ( io_code, 6, 4, 1, numb, regset );
                          compact_store_matrix_gen_ ( io_code, 0, ldb, i, j+1, 6, numb, datasz, regset, breg );
                       }
                       if ( j+2 <= n1 ) {
                          compact_load_matrix_gen_ ( io_code, 0, ldb, i, j+2, 8, numb, datasz, regset, breg );
                          compact_fma_cplusab_ ( io_code, 8, 7, 1, numb, regset );
                          compact_store_matrix_gen_ ( io_code, 0, ldb, i, j+2, 8, numb, datasz, regset, breg );
                       }
                    }
                    if ( nounit ) {
                       compact_load_matrix_gen_ ( io_code, 0, lda, k, k, 1, numb, datasz, regset, areg );
                       compact_mult_two_nums_ ( io_code, 0, 1, 0, numb, regset );
                       if ( j+1 <= n1 ) {
                          compact_mult_two_nums_ ( io_code, 4, 1, 4, numb, regset );
                       }
                       if ( j+2 <= n1 ) {
                          compact_mult_two_nums_ ( io_code, 7, 1, 7, numb, regset );
                       }
                    }
                    compact_store_matrix_gen_ ( io_code, 0, ldb, k, j, 0, numb, datasz, regset, breg );
                    if ( j+1 <= n1 ) {
                       compact_store_matrix_gen_ ( io_code, 0, ldb, k, j+1, 4, numb, datasz, regset, breg );
                    }
                    if ( j+2 <= n1 ) {
                       compact_store_matrix_gen_ ( io_code, 0, ldb, k, j+2, 7, numb, datasz, regset, breg );
                    }
                 }
              }
           } else {
              /* Do LLN* cases: B <- alpha * inv(A)*B */
              for ( j = 1; j <= n1; j+=3 )
              {
                 for ( k = m1; k >= 1; k-- )
                 {
                    compact_load_matrix_gen_ ( io_code, 0, ldb, k, j, 0, numb, datasz, regset, breg );
                    if ( j+1<=n1 ) compact_load_matrix_gen_ ( io_code, 0, ldb, k, j+1, 4, numb, datasz, regset, breg );
                    if ( j+2<=n1 ) compact_load_matrix_gen_ ( io_code, 0, ldb, k, j+2, 7, numb, datasz, regset, breg );
                    if ( LIBXSMM_NEQ(1, alpha) ) {
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, k, j, 0, numb, datasz, regset, breg );
                       if ( j+1 <= n1 ) {
                          compact_mult_two_nums_ ( io_code, 4, 2, 4, numb, regset );
                          compact_store_matrix_gen_ ( io_code, 0, ldb, k, j+1, 4, numb, datasz, regset, breg );
                       }
                       if ( j+2 <= n1 ) {
                          compact_mult_two_nums_ ( io_code, 7, 2, 7, numb, regset );
                          compact_store_matrix_gen_ ( io_code, 0, ldb, k, j+2, 7, numb, datasz, regset, breg );
                       }
                    }
                    if ( nounit ) {
                       compact_load_matrix_gen_ ( io_code, 0, lda, k, k, 1, numb, datasz, regset, areg );
                       compact_mult_two_nums_ ( io_code, 0, 1, 3, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, k, j, 3, numb, datasz, regset, breg );
                       if ( j+1 <= n1 ) {
                          compact_mult_two_nums_ ( io_code, 4, 1, 6, numb, regset );
                          compact_store_matrix_gen_ ( io_code, 0, ldb, k, j+1, 6, numb, datasz, regset, breg );
                       }
                       if ( j+2 <= n1 ) {
                          compact_mult_two_nums_ ( io_code, 7, 1, 8, numb, regset );
                          compact_store_matrix_gen_ ( io_code, 0, ldb, k, j+2, 8, numb, datasz, regset, breg );
                       }
                    }
                    for ( i = k+1; i <= m1; i++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, j, 3, numb, datasz, regset, breg );
                       compact_load_matrix_gen_ ( io_code, 0, lda, i, k, 1, numb, datasz, regset, areg );
                       compact_fma_cplusab_ ( io_code, 3, 0, 1, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 3, numb, datasz, regset, breg );
                       if ( j+1 <= n1 ) {
                          compact_load_matrix_gen_ ( io_code, 0, ldb, i, j+1, 6, numb, datasz, regset, breg);
                          compact_fma_cplusab_ ( io_code, 6, 4, 1, numb, regset );
                          compact_store_matrix_gen_ ( io_code, 0, ldb, i, j+1, 6, numb, datasz, regset, breg );
                       }
                       if ( j+2 <= n1 ) {
                          compact_load_matrix_gen_ ( io_code, 0, ldb, i, j+2, 8, numb, datasz, regset, breg);
                          compact_fma_cplusab_ ( io_code, 8, 7, 1, numb, regset );
                          compact_store_matrix_gen_ ( io_code, 0, ldb, i, j+2, 8, numb, datasz, regset, breg );
                       }
                    } /* for i LLN main loop */
                 }    /* for k LLN loop */
              }       /* for j LLN loop */
           } /* uplo */
        } else {
           if ( (uplo=='U') || (uplo=='u') )
           {
              /* Do LUT* cases: B<- alpha*A^T*B */
              for ( j = 1; j <= n1; j+=3 )
              {
                 for ( i = m1; i >= 1; i-- ) {
                    compact_load_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                    if ( j+1 <= n1 ) compact_load_matrix_gen_ ( io_code, 0, ldb, i, j+1, 4, numb, datasz, regset, breg );
                    if ( j+2 <= n1 ) compact_load_matrix_gen_ ( io_code, 0, ldb, i, j+2, 7, numb, datasz, regset, breg );
                    if ( nounit ) {
                       compact_load_matrix_gen_ ( io_code, 0, lda, i, i, 1, numb, datasz, regset, areg );
                       compact_mult_two_nums_ ( io_code, 0, 1, 0, numb, regset );
                       if ( j+1 <= n1 ) compact_mult_two_nums_ ( io_code, 4, 1, 4, numb, regset );
                       if ( j+2 <= n1 ) compact_mult_two_nums_ ( io_code, 7, 1, 7, numb, regset );
                    }
                    for ( k = 1; k <= i-1; k++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, k, j, 3, numb, datasz, regset, breg );
                       compact_load_matrix_gen_ ( io_code, 0, lda, k, i, 1, numb, datasz, regset, areg );
                       compact_fma_cplusab_ ( io_code, 0, 1, 3, numb, regset );
                       if ( j+1 <= n1 ) {
                          compact_load_matrix_gen_ ( io_code, 0, ldb, k, j+1, 6, numb, datasz, regset, breg );
                          compact_fma_cplusab_ ( io_code, 4, 1, 6, numb, regset );
                       }
                       if ( j+2 <= n1 ) {
                          compact_load_matrix_gen_ ( io_code, 0, ldb, k, j+2, 8, numb, datasz, regset, breg );
                          compact_fma_cplusab_ ( io_code, 7, 1, 8, numb, regset );
                       }
                    }
                    if ( LIBXSMM_NEQ(1, alpha) ) {
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
                       if ( j+1 <= n1 ) compact_mult_two_nums_ ( io_code, 4, 2, 4, numb, regset );
                       if ( j+2 <= n1 ) compact_mult_two_nums_ ( io_code, 7, 2, 7, numb, regset );
                    }
                    compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                    if ( j+1 <= n1 ) compact_store_matrix_gen_ ( io_code, 0, ldb, i, j+1, 4, numb, datasz, regset, breg );
                    if ( j+2 <= n1 ) compact_store_matrix_gen_ ( io_code, 0, ldb, i, j+2, 7, numb, datasz, regset, breg );
                 }
              }
           } else {
              /* Do LLT* cases: B <- alpha * A*B */
              for ( j = 1; j <= n1; j+=3 )
              {
                 for ( i = 1; i <= m1; i++ ) {
                    compact_load_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                    if ( j+1 <= n1 ) compact_load_matrix_gen_ ( io_code, 0, ldb, i, j+1, 4, numb, datasz, regset, breg );
                    if ( j+2 <= n1 ) compact_load_matrix_gen_ ( io_code, 0, ldb, i, j+2, 7, numb, datasz, regset, breg );
                    if ( nounit ) {
                       compact_load_matrix_gen_ ( io_code, 0, lda, i, i, 1, numb, datasz, regset, areg );
                       compact_mult_two_nums_ ( io_code, 0, 1, 0, numb, regset );
                       if ( j+1 <= n1 ) compact_mult_two_nums_ ( io_code, 4, 1, 4, numb, regset );
                       if ( j+2 <= n1 ) compact_mult_two_nums_ ( io_code, 7, 1, 7, numb, regset );
                    }
                    for ( k = i+1; k <= m1; k++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, k, j, 3, numb, datasz, regset, breg );
                       compact_load_matrix_gen_ ( io_code, 0, lda, k, i, 1, numb, datasz, regset, areg );
                       compact_fma_cplusab_ ( io_code, 0, 1, 3, numb, regset );
                       if ( j+1 <= n1 ) {
                          compact_load_matrix_gen_ ( io_code, 0, ldb, k, j+1, 6, numb, datasz, regset, breg );
                          compact_fma_cplusab_ ( io_code, 4, 1, 6, numb, regset );
                       }
                       if ( j+2 <= n1 ) {
                          compact_load_matrix_gen_ ( io_code, 0, ldb, k, j+2, 8, numb, datasz, regset, breg );
                          compact_fma_cplusab_ ( io_code, 7, 1, 8, numb, regset );
                       }
                    }
                    if ( LIBXSMM_NEQ(1, alpha) ) {
                       compact_mult_two_nums_ ( io_code, 0, 2, 0, numb, regset );
                       if ( j+1 <= n1 ) compact_mult_two_nums_ ( io_code, 4, 2, 4, numb, regset );
                       if ( j+2 <= n1 ) compact_mult_two_nums_ ( io_code, 7, 2, 7, numb, regset );
                    }
                    compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                    if ( j+1 <= n1 ) compact_store_matrix_gen_ ( io_code, 0, ldb, i, j+1, 4, numb, datasz, regset, breg );
                    if ( j+2 <= n1 ) compact_store_matrix_gen_ ( io_code, 0, ldb, i, j+2, 7, numb, datasz, regset, breg );
                 }
              }
           } /* uplo */
        } /* trans */
     } else {
        if ( (trans=='N') || (trans=='n') )
        {
           if ( (uplo=='U') || (uplo=='u') )
           {
              /* Do RUN* cases: B<- alpha*B*A */
              for ( j = n1; j >= 1; j-- ) {
                 if ( nounit ) {
                    compact_load_matrix_gen_ ( io_code, 0, lda, j, j, 1, numb, datasz, regset, areg );
                 }
                 if ( LIBXSMM_NEQ(1, alpha) ) {
                    if ( nounit ) {
                       compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
                    }
#ifdef GENERATOR_PACKED_TRMM_DEBUG
                    else {
                       printf("wrong temp values for TRMM's RUN\n");
                    }
#endif
                 }
                 if ( LIBXSMM_NEQ(1, alpha) || nounit ) {
                    for ( i = 1; i <= m1; i++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                       compact_mult_two_nums_ ( io_code, 0, 1, 0, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                    }
                 }
                 for ( k = 1; k <= j - 1; k++ ) {
                    compact_load_matrix_gen_ ( io_code, 0, lda, k, j, 1, numb, datasz, regset, areg );
                    if ( LIBXSMM_NEQ(1, alpha) ) {
                       compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
                    }
                    for ( i = 1; i <= m1; i++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, k, 3, numb, datasz, regset, breg );
                       compact_fma_cplusab_ ( io_code, 0, 1, 3, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                    }
                 }
              }
           } else {
              /* Do RLN* cases: B <- alpha * B * A */
              for ( j = 1; j <= n1; j++ )
              {
                 if ( nounit ) {
                    compact_load_matrix_gen_ ( io_code, 0, lda, j, j, 1, numb, datasz, regset, areg );
                 }
                 if ( LIBXSMM_NEQ(1, alpha) ) {
                    if ( nounit ) {
                       compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
                    }
#ifdef GENERATOR_PACKED_TRMM_DEBUG
                    else {
                       printf("wrong temp values for TRMM's RLN\n");
                    }
#endif
                 }
                 if ( LIBXSMM_NEQ(1, alpha) || nounit ) {
                    for ( i = 1; i <= m1; i++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                       compact_mult_two_nums_ ( io_code, 0, 1, 0, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                    }
                 }
                 for ( k = j+1; k <= n1; k++ ) {
                    compact_load_matrix_gen_ ( io_code, 0, lda, k, j, 1, numb, datasz, regset, areg );
                    if ( LIBXSMM_NEQ(1, alpha) ) {
                       compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
                    }
                    for ( i = 1; i <= m1; i++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, k, 3, numb, datasz, regset, breg );
                       compact_fma_cplusab_ ( io_code, 0, 1, 3, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                    }
                 }
              }
           } /* uplo */
        } else {
           if ( (uplo=='U') || (uplo=='u') )
           {
              /* Do RUT* cases: B<- alpha*B *A^T */
              for ( k = 1; k <= n1; k++ )
              {
                 for ( j = 1; j <= k-1; j++ )
                 {
                    compact_load_matrix_gen_ ( io_code, 0, lda, j, k, 1, numb, datasz, regset, areg );
                    if ( LIBXSMM_NEQ(1, alpha) ) {
                       compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
                    }
                    for ( i = 1; i <= m1; i++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, k, 3, numb, datasz, regset, breg );
                       compact_fma_cplusab_ ( io_code, 0, 1, 3, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                    }
                 }
                 if ( nounit ) {
                    compact_load_matrix_gen_ ( io_code, 0, lda, k, k, 1, numb, datasz, regset, areg );
                 }
                 if ( LIBXSMM_NEQ(1, alpha) ) {
                    if ( nounit ) {
                       compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
                    }
#ifdef GENERATOR_PACKED_TRMM_DEBUG
                    else {
                       printf("wrong temp values for TRMM's RUT\n");
                    }
#endif
                 }
                 if ( LIBXSMM_NEQ(1, alpha) || nounit ) {
                    for ( i = 1; i <= m1; i++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, k, 0, numb, datasz, regset, breg );
                       compact_mult_two_nums_ ( io_code, 0, 1, 0, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, i, k, 0, numb, datasz, regset, breg );
                    }
                 }
              }
           } else {
              /* Do RLT* cases: B <- alpha * B *inv(A^T) */
              for ( k = n1; k >= 1; k-- )
              {
                 for ( j = k+1; j <= n1; j++ )
                 {
                    compact_load_matrix_gen_ ( io_code, 0, lda, j, k, 1, numb, datasz, regset, areg );
                    if ( LIBXSMM_NEQ(1, alpha) ) {
                       compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
                    }
                    for ( i = 1; i <= m1; i++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, k, 3, numb, datasz, regset, breg );
                       compact_fma_cplusab_ ( io_code, 0, 1, 3, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, i, j, 0, numb, datasz, regset, breg );
                    }
                 }
                 if ( nounit ) {
                    compact_load_matrix_gen_ ( io_code, 0, lda, k, k, 1, numb, datasz, regset, areg );
                 }
                 if ( LIBXSMM_NEQ(1, alpha) ) {
                    if ( nounit ) {
                       compact_mult_two_nums_ ( io_code, 1, 2, 1, numb, regset );
                    }
#ifdef GENERATOR_PACKED_TRMM_DEBUG
                    else {
                       printf("wrong temp values for TRMM's RLT\n");
                    }
#endif
                 }
                 if ( LIBXSMM_NEQ(1, alpha) || nounit ) {
                    for ( i = 1; i <= m1; i++ ) {
                       compact_load_matrix_gen_ ( io_code, 0, ldb, i, k, 0, numb, datasz, regset, breg );
                       compact_mult_two_nums_ ( io_code, 0, 1, 0, numb, regset );
                       compact_store_matrix_gen_ ( io_code, 0, ldb, i, k, 0, numb, datasz, regset, breg );
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
