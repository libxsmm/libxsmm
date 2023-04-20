/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_common.h"
#include "generator_gemm_common.h"
#include "generator_gemm_sse_avx_avx2_avx512.h"
#include "generator_gemm_amx.h"
#include "generator_gemm_amx_emu.h"
#include "generator_gemm_aarch64.h"
#include "generator_gemm_noarch.h"


LIBXSMM_API
void libxsmm_generator_gemm_kernel( libxsmm_generated_code*        io_generated_code,
                                    const libxsmm_gemm_descriptor* i_xgemm_desc ) {
  /* apply the alignment override */
  libxsmm_gemm_descriptor l_xgemm_desc_mod = *i_xgemm_desc;
  unsigned int l_vector_length = 1;
  int l_emu_amx = 0;
  int l_aarch64_bfdot = libxsmm_cpuid_arm_use_bfdot();
  const char *const l_env_emu_amx = getenv("EMULATE_AMX");
  unsigned int l_saved_arch = io_generated_code->arch;
  if ( 0 == l_env_emu_amx ) {
  } else {
    l_emu_amx = atoi(l_env_emu_amx);
  }

  /* We allow b vnniT for x86 and bf16 whenever possible */
  if ( ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0) && ( io_generated_code->arch >= LIBXSMM_X86_GENERIC ) &&  ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) ) {
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
      if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) ) {
        /* we are fine */
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
      }
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
      return;
    }
  }

  /* for low precision lets make KNL/KNM an AVX2 machine */
  if ( ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_KNM ) || ( io_generated_code->arch == LIBXSMM_X86_AVX512_MIC ) ) &&
       ( ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ||
         ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) ) {
    io_generated_code->arch = LIBXSMM_X86_AVX2;
  }
  if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_KNM ) &&
       ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) &&
       ( l_xgemm_desc_mod.k % 8 != 0) ) {
    io_generated_code->arch = LIBXSMM_X86_AVX2;
  }
  if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_MIC ) &&
       ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
    io_generated_code->arch = LIBXSMM_X86_AVX2;
  }

  /* overwrite VNNI Flag when K == 1 */
  if ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) &&
       (l_xgemm_desc_mod.k == 1) &&
       ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
    l_xgemm_desc_mod.flags = l_xgemm_desc_mod.flags & (~LIBXSMM_GEMM_FLAG_VNNI_A);
  }

  if ( (LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_OUT( l_xgemm_desc_mod.datatype )) ||
       (LIBXSMM_DATATYPE_I8  == LIBXSMM_GETENUM_OUT( l_xgemm_desc_mod.datatype )) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  /* determining vector length depending on architecture and precision */
  if ( io_generated_code->arch <= LIBXSMM_TARGET_ARCH_GENERIC ) {
    /* nothing todo */
  } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 2;
  } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 8;
  } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX2 ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128 ) && LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )  ) {
    l_vector_length = 8;
    if (l_xgemm_desc_mod.k % 4 != 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
  } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX2 ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128 ) && ( (LIBXSMM_DATATYPE_I16  == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) ) ) {
    l_vector_length = 8;
    if (l_xgemm_desc_mod.k % 2 != 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
  } else if ( ( io_generated_code->arch >= LIBXSMM_X86_AVX2 ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512_VL128 ) && ( (LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) ) ) {
    /* some checks as we cannot mask everything */
    if ( (l_xgemm_desc_mod.k % 2 != 0) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
      l_vector_length = 16;
    } else {
      l_vector_length = 8;
    }
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_AVX512_VL256 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_AVX512_VL256 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 8;
  } else if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CLX ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 8;
  } else if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_VL256_CPX ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 8;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 8;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 16;
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512 ) &&
              ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
    l_vector_length = 8;
    /* some checks as we cannot mask everything */
    if (l_xgemm_desc_mod.k % 2 != 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
  } else if ( ( io_generated_code->arch == LIBXSMM_X86_AVX512_KNM ) &&
              ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
    l_vector_length = 16;
    /* some checks as we cannot mask everything */
    if ( (l_xgemm_desc_mod.k % 8 != 0) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE ) &&
              ( LIBXSMM_DATATYPE_I16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
    l_vector_length = 16;
    /* some checks as we cannot mask everything */
    if (l_xgemm_desc_mod.k % 2 != 0) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512 ) &&
              ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ||
                ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ||
                ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) )    ) ) {
    l_vector_length = 8;
    if ( (l_xgemm_desc_mod.k % 4 != 0) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE ) &&
              ( ( LIBXSMM_DATATYPE_I8  == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ||
                ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP(  l_xgemm_desc_mod.datatype ) )  ||
                ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP(  l_xgemm_desc_mod.datatype ) ) ) ) {
    l_vector_length = 16;
    /* some checks as we cannot mask everything */
    if ( (l_xgemm_desc_mod.k % 4 != 0) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( io_generated_code->arch >= LIBXSMM_X86_AVX512_VL256 ) && ( io_generated_code->arch < LIBXSMM_X86_AVX512 ) &&
              ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
    /* some checks as we cannot mask everything */
    if ( (l_xgemm_desc_mod.k % 2 != 0) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
      l_vector_length = 16;
      /*l_xgemm_desc_mod.k = l_xgemm_desc_mod.k;*/
      /*l_xgemm_desc_mod.ldb = l_xgemm_desc_mod.ldb;*/
    } else {
      l_vector_length = 8;
    }
  } else if ( ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) &&
              ( io_generated_code->arch >= LIBXSMM_X86_AVX512_CORE ) &&
              ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
    /* some checks as we cannot mask everything */
    if ( (l_xgemm_desc_mod.k % 2 != 0) && ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
      l_vector_length = 32;
      /*l_xgemm_desc_mod.k = l_xgemm_desc_mod.k;*/
      /*l_xgemm_desc_mod.ldb = l_xgemm_desc_mod.ldb;*/
    } else {
      l_vector_length = 16;
    }
  } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_V81 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_V81 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 2;
  } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_V82 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_V82 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 2;
  } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 2;
  } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 8;
  } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 4;
  } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE512 || io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) && LIBXSMM_DATATYPE_F32 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 16;
  } else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE512 || io_generated_code->arch == LIBXSMM_AARCH64_A64FX ) && LIBXSMM_DATATYPE_F64 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
    l_vector_length = 8;
  } else if ( ( io_generated_code->arch >= LIBXSMM_AARCH64_V81    )  &&
              ( io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT ) &&
              (    ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) )
                || ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) &&
              ( l_aarch64_bfdot != 0 ) ) {
    /* TODO (BFDOT): add flags and check on MMLA-formated A/B */
    /* TODO (BFDOT): add support for at least m % 2 == 0, k % 4 == 0 when running BF16 */
    /* TODO (BFDOT): adjust checks for future SVE kernels */
    if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
      if (l_xgemm_desc_mod.k % 2 != 0) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
        return;
      }
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
      if (l_xgemm_desc_mod.k % 4 != 0)  {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
        return;
      }
    }
    /* ASIMD + BFDOT */
    if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE128 ) {
      l_vector_length = 4;
    }
    /* SVE256 + BFDOT */
    else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 )  ) {
      l_vector_length = 8;
    }
    else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
      return;
    }
  } else if ( ( io_generated_code->arch >= LIBXSMM_AARCH64_V81    )  &&
              ( io_generated_code->arch <= LIBXSMM_AARCH64_ALLFEAT ) &&
              (    ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) )
                || ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) &&
              ( l_aarch64_bfdot == 0 ) ) {
    /* TODO (MMLA): add flags and check on MMLA-formated A/B */
    /* TODO (MMLA): add support for at least m % 2 == 0, k % 4 == 0 when running BF16 */
    /* TODO (MMLA): adjust checks for future SVE kernels */
    if ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
      if (l_xgemm_desc_mod.k % 4 != 0) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
        return;
      }
    } else if ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) {
      if (l_xgemm_desc_mod.k % 8 != 0)  {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
        return;
      }
    }
    /* ASIMD + MMLA */
    /* TODO: These are not properly implemented yet */
    if ( io_generated_code->arch < LIBXSMM_AARCH64_SVE128 ) {
#if 0
      l_vector_length = 4;
#else
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
#endif
    }
    /* SVE256 + MMLA */
    else if ( ( io_generated_code->arch == LIBXSMM_AARCH64_SVE256 || io_generated_code->arch == LIBXSMM_AARCH64_NEOV1 )  ) {
      l_vector_length = 8;
    }
    else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
      return;
    }
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
      return;
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH_PREC );
    return;
  }

  /* check LDA */
  if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_A) == LIBXSMM_GEMM_FLAG_TRANS_A ) {
    if ( l_xgemm_desc_mod.lda < l_xgemm_desc_mod.k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA_TRANS );
      return;
    }
    if ((LIBXSMM_DATATYPE_F32 != LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) &&
        (LIBXSMM_DATATYPE_F64 != LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return;
    }
  } else {
    if ( l_xgemm_desc_mod.lda < l_xgemm_desc_mod.m ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDA );
      return;
    }
  }

  /* check LDB */
  if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    if ( l_xgemm_desc_mod.ldb < l_xgemm_desc_mod.n ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB_TRANS );
      return;
    }
  } else {
    if ( l_xgemm_desc_mod.ldb < l_xgemm_desc_mod.k ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDB );
      return;
    }
  }

  /* check LDC */
  if ( l_xgemm_desc_mod.ldc < l_xgemm_desc_mod.m ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_LDC );
    return;
  }

  /* check for trans A cases which are not supported in the generator */
  if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_A) > 0 ) {
    if ( (LIBXSMM_DATATYPE_F32 != LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) &&
         (LIBXSMM_DATATYPE_F64 != LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_A );
      return;
    } else {
      /* we are fine, we have transpose support */
    }
  }

  /* check for trans B cases which are not supported in the generator */
  if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 ) {
    if ( (LIBXSMM_DATATYPE_I16  == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) ||
         (LIBXSMM_DATATYPE_I8   == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ))    ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
      return;
    } else {
      /* we are fine, we have transpose support */
    }
    if ( ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
      if ( (l_aarch64_bfdot == 0 ) && ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 )) {
        /* we are fine, we do support mmla kernels with B in vnni4t */
      } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_TRANS_B );
        return;
      }
    }
  }

  if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 ) {
    if ( (l_aarch64_bfdot == 0 ) && ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_TRANS_B) > 0 )) {
      /* we are fine, we do support mmla kernels with B in vnni4t */
    } else if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) > 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
      return;
    }
  }

  /* check for VNNI flag being set in case of low precision GEMM */
  /* TODO (MMLA): adjust for aarch64 using i8-MMLA instructions
  if ( ( LIBXSMM_DATATYPE_I16  == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ||
       ( LIBXSMM_DATATYPE_I8   == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
      return;
    }
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) == 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_A );
      return;
    }
  }
  if ( ( LIBXSMM_DATATYPE_BF16  == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) {
    if ( (l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_B) > 0 ) {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_VNNI_B );
      return;
    }
  }
  */

  /* right now we only support eltwise fusion on SPR and BF16 */
  /* TODO: EVANGELOS -- AMMEND */
#if 0
  if ( ( (io_generated_code->arch < LIBXSMM_X86_AVX512_SPR) || (LIBXSMM_DATATYPE_BF16 != LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) ) &&
       ( l_xgemm_desc_mod.meltw_operation != LIBXSMM_MELTW_OPERATION_NONE ) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }
#endif

  /* check if alignment is not possible */
  if ( 0 != (l_xgemm_desc_mod.lda % l_vector_length) ) {
    l_xgemm_desc_mod.flags &= ~LIBXSMM_GEMM_FLAG_ALIGN_A;
  }
  if ( 0 != (l_xgemm_desc_mod.ldc % l_vector_length) ) {
    l_xgemm_desc_mod.flags &= ~LIBXSMM_GEMM_FLAG_ALIGN_C;
  }

  if ( io_generated_code->arch <= LIBXSMM_TARGET_ARCH_GENERIC ) {
    libxsmm_generator_gemm_noarch_kernel( io_generated_code, &l_xgemm_desc_mod );
  } else if ( io_generated_code->arch <= LIBXSMM_X86_ALLFEAT ) {
    /* call actual kernel generation with revised parameters */
    /* TODO: check for VNNI format */
    if ( ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR ) && ( io_generated_code->arch < LIBXSMM_X86_ALLFEAT ) ) &&
         ( ( LIBXSMM_DATATYPE_BF16 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ||
           ( LIBXSMM_DATATYPE_I8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype ) ) ) &&
         ((l_xgemm_desc_mod.flags & LIBXSMM_GEMM_FLAG_VNNI_A) != 0) ) {
      if (l_emu_amx == 0) {
        libxsmm_generator_gemm_amx_kernel_wrapper( io_generated_code, &l_xgemm_desc_mod );
      } else {
        /* let's recheck CPU to even emulation AVX512_BF16 */
        io_generated_code->arch = libxsmm_cpuid(NULL);
        l_xgemm_desc_mod.c3 = 0;
        libxsmm_generator_gemm_amx_kernel_emu_wrapper( io_generated_code, &l_xgemm_desc_mod );
      }
    } else if ( ( ( io_generated_code->arch >= LIBXSMM_X86_AVX512_SPR ) && ( io_generated_code->arch < LIBXSMM_X86_ALLFEAT ) ) &&
                ( ( LIBXSMM_DATATYPE_BF8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )) || ( LIBXSMM_DATATYPE_HF8 == LIBXSMM_GETENUM_INP( l_xgemm_desc_mod.datatype )))) {
      libxsmm_generator_gemm_amx_kernel_wrapper( io_generated_code, &l_xgemm_desc_mod );
    } else {
      if (l_emu_amx != 0) {
        io_generated_code->arch = libxsmm_cpuid(NULL);
      }
      libxsmm_generator_gemm_sse_avx_avx2_avx512_kernel_wrapper( io_generated_code, &l_xgemm_desc_mod );
    }
  } else if ( (io_generated_code->arch == LIBXSMM_AARCH64_V81) || (io_generated_code->arch == LIBXSMM_AARCH64_V82) ) {
    libxsmm_generator_gemm_aarch64_kernel( io_generated_code, &l_xgemm_desc_mod );
  } else if ( io_generated_code->arch == LIBXSMM_AARCH64_APPL_M1 ) {
    libxsmm_generator_gemm_aarch64_kernel( io_generated_code, &l_xgemm_desc_mod );
  } else if ( (io_generated_code->arch == LIBXSMM_AARCH64_SVE256) || (io_generated_code->arch == LIBXSMM_AARCH64_NEOV1) ) {
    libxsmm_generator_gemm_aarch64_kernel( io_generated_code, &l_xgemm_desc_mod );
  } else if ( (io_generated_code->arch == LIBXSMM_AARCH64_SVE512) || (io_generated_code->arch == LIBXSMM_AARCH64_A64FX) ) {
    libxsmm_generator_gemm_aarch64_kernel( io_generated_code, &l_xgemm_desc_mod );
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_ARCH );
    return;
  }

  /* restore arch */
  io_generated_code->arch = l_saved_arch;
}


LIBXSMM_API
void libxsmm_generator_gemm_inlineasm( const char*                    i_file_out,
                                       const char*                    i_routine_name,
                                       const libxsmm_gemm_descriptor* i_xgemm_desc,
                                       const char*                    i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code /*= { 0 }*/;
  LIBXSMM_MEMZERO127(&l_generated_code);

  /* set arch */
  if ( strcmp(i_arch, "wsm") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_SSE42;
  } else if ( strcmp(i_arch, "snb") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX;
  } else if ( strcmp(i_arch, "hsw") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX2;
  } else if ( strcmp(i_arch, "adl") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX2_ADL;
  } else if ( strcmp(i_arch, "srf") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX2_SRF;
  } else if ( strcmp(i_arch, "knl") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_MIC;
  } else if ( strcmp(i_arch, "knm") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_KNM;
  } else if ( strcmp(i_arch, "skx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CORE;
  } else if ( strcmp(i_arch, "clx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CLX;
  } else if ( strcmp(i_arch, "cpx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CPX;
  } else if ( strcmp(i_arch, "spr") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_SPR;
  } else if ( strcmp(i_arch, "gnr") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_GNR;
  } else if ( strcmp(i_arch, "avx512_vl256") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_VL256;
  } else if ( strcmp(i_arch, "avx512_vl256_clx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_VL256_CLX;
  } else if ( strcmp(i_arch, "avx512_vl256_cpx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_VL256_CPX;
  } else {
    l_generated_code.arch = LIBXSMM_X86_GENERIC;
  }

  /* add signature to code string */
  libxsmm_mmfunction_signature( &l_generated_code, i_routine_name, i_xgemm_desc );

  /* add instruction set mismatch check to code, header */
  libxsmm_generator_isa_check_header( &l_generated_code );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_gemm_kernel( &l_generated_code, i_xgemm_desc );

  /* add instruction set mismatch check to code, footer */
  libxsmm_generator_isa_check_footer( &l_generated_code );

  /* add flop counter for debug compilation */
  libxsmm_generator_gemm_add_flop_counter( &l_generated_code, i_xgemm_desc );

  /* close current function */
  libxsmm_close_function( &l_generated_code );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    LIBXSMM_HANDLE_ERROR_VERBOSE( &l_generated_code, l_generated_code.last_error );
    return;
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "a" );
    if ( l_file_handle != NULL ) {
      assert(l_generated_code.generated_code != NULL);
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR libxsmm_generator_gemm_inlineasm could not write to into destination source file\n");
      return;
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}


LIBXSMM_API
void libxsmm_generator_gemm_directasm(const char*                     i_file_out,
                                      const char*                     i_routine_name,
                                      const libxsmm_gemm_descriptor* i_xgemm_desc,
                                      const char*                     i_arch ) {
  /* init generated code object */
  libxsmm_generated_code l_generated_code /*= { 0 }*/;
  l_generated_code.generated_code = NULL;
  l_generated_code.buffer_size = 0;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 1;
  l_generated_code.last_error = 0;
  l_generated_code.arch = 0;
  l_generated_code.sf_size = 0;

  /* set arch */
  if ( strcmp(i_arch, "wsm") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_SSE42;
  } else if ( strcmp(i_arch, "snb") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX;
  } else if ( strcmp(i_arch, "hsw") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX2;
  } else if ( strcmp(i_arch, "adl") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX2_ADL;
  } else if ( strcmp(i_arch, "srf") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX2_SRF;
  } else if ( strcmp(i_arch, "knl") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_MIC;
  } else if ( strcmp(i_arch, "knm") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_KNM;
  } else if ( strcmp(i_arch, "skx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CORE;
  } else if ( strcmp(i_arch, "clx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CLX;
  } else if ( strcmp(i_arch, "cpx") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_CPX;
  } else if ( strcmp(i_arch, "spr") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_SPR;
  } else if ( strcmp(i_arch, "gnr") == 0  ) {
    l_generated_code.arch = LIBXSMM_X86_AVX512_GNR;
  } else {
    l_generated_code.arch = LIBXSMM_X86_GENERIC;
  }

  /* check if we are not noarch */
  if ( strcmp( i_arch, "noarch" ) == 0 ) {
    fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_gemm_direct: we cannot create ASM when noarch is specified!\n");
    return;
  }

  /* add signature to code string */
  libxsmm_mmfunction_signature( &l_generated_code, i_routine_name, i_xgemm_desc );

  /* generate the actual kernel code for current description depending on the architecture */
  libxsmm_generator_gemm_kernel( &l_generated_code, i_xgemm_desc );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    LIBXSMM_HANDLE_ERROR_VERBOSE( &l_generated_code, l_generated_code.last_error );
    return;
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( i_file_out, "w" );
    if ( l_file_handle != NULL ) {
      assert(l_generated_code.generated_code != NULL);
      fputs( (const char*)l_generated_code.generated_code, l_file_handle );
      fclose( l_file_handle );
    } else {
      fprintf(stderr, "LIBXSMM ERROR, libxsmm_generator_gemm_direct: could not write to into destination source file!\n");
      return;
    }
  }

  /* free code memory */
  free( l_generated_code.generated_code );
}

