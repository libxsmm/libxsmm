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
#include <libxsmm_utils.h>
#include <libxsmm.h>
#include <float.h>

#define OP_NONE         0
#define COLBIAS_ADD     1
#define RELU_NOBITMASK  1
#define RELU_BITMASK    2
#define SIGMOID         3

#define EXPORT_CODE_BUFFERSIZE 262144

LIBXSMM_INLINE
libxsmm_datatype char_to_libxsmm_datatype( const char* dt ) {
  libxsmm_datatype dtype = LIBXSMM_DATATYPE_UNSUPPORTED;

  if ( (strcmp(dt, "F64") == 0) ) {
    dtype = LIBXSMM_DATATYPE_F64;
  } else if ( (strcmp(dt, "I64") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I64;
  } else if ( (strcmp(dt, "F32") == 0) ) {
    dtype = LIBXSMM_DATATYPE_F32;
  } else if ( (strcmp(dt, "I32") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I32;
  } else if ( (strcmp(dt, "F16") == 0) ) {
    dtype = LIBXSMM_DATATYPE_F16;
  } else if ( (strcmp(dt, "BF16") == 0) ) {
    dtype = LIBXSMM_DATATYPE_BF16;
  } else if ( (strcmp(dt, "I16") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I16;
  } else if ( (strcmp(dt, "BF8") == 0) ) {
    dtype = LIBXSMM_DATATYPE_BF8;
  } else if ( (strcmp(dt, "HF8") == 0) ) {
    dtype = LIBXSMM_DATATYPE_HF8;
  } else if ( (strcmp(dt, "I8") == 0) ) {
    dtype = LIBXSMM_DATATYPE_I8;
  } else if ( (strcmp(dt, "U8") == 0) ) {
    dtype = LIBXSMM_DATATYPE_U8;
  } else if ( (strcmp(dt, "IMPLICIT") == 0) ) {
    dtype = LIBXSMM_DATATYPE_IMPLICIT;
  } else {
    dtype = LIBXSMM_DATATYPE_UNSUPPORTED;
  }

  return dtype;
}

LIBXSMM_INLINE
void print_help(void) {
  printf("\n\n");
  printf("Arch, file, kernel:\n");
  printf("    arch (wsm,snb,hsw,adl,srf,clx,cpx,spr,gnr)\n");
  printf("    export-filename\n");
  printf("    kernel (gemm, gemm-ext, eltwise)\n\n");
  printf("gemm:\n");
  printf("    A Precision (I8, U8, I16, BF8, HF8, BF16, F32, F64)\n");
  printf("    B Precision (I8, U8, I16, BF8, HF8, BF16, F32, F64)\n");
  printf("    Compute Precision (I32, F32, F64)\n");
  printf("    C Precision (I32, BF8, HF8, BF16, F32, F64)\n");
  printf("    M\n");
  printf("    N\n");
  printf("    K\n");
  printf("    LDA\n");
  printf("    LDB\n");
  printf("    LDC\n");
  printf("    alpha: 1\n");
  printf("    beta: 0 or 1\n");
  printf("    0: unaligned A, otherwise aligned\n");
  printf("    0: unaligned C, otherwise aligned\n");
  printf("    0: A normal, 1: A trans\n");
  printf("    0: B normal, 1: B trans\n");
  printf("    0: A normal, 1: A vnni\n");
  printf("    0: B normal, 1: B vnni\n");
  printf("    0: C normal, 1: C vnni\n");
  printf("    PREFETCH: nopf (none), BL2viaC, AL2, curAL2, AL2_BL2viaC, curAL2_BL2viaC\n");
  printf("    BRGEMM/SPMM: nobr, addrbr, offsbr, strdbr, spmm\n");
  printf("    BRsize: 1 - N (or sparsity factor for spmm)\n");
  printf("    BRunroll: 0/1\n");
  printf("    BR-stride-A (ignorded if not strdbr)\n");
  printf("    BR-stride-B (ignorded if not strdbr)\n");
  printf("    tile configuration: 1 - external, 0 - internal\n");
  printf("gemm-ext (extra parameter)\n");
  printf("    post_gemm_binary: 0 - none, 1 - colbias_add\n");
  printf("    post_gemm_binday_LD\n");
  printf("    post_gemm_unary: 0 - none, 1 - relu_nobitmask, 2 - relu_bitmask, 3 - sigmoid \n");
  printf("    post_gemm_unary_LD\n");
  printf("\n\n");
}

LIBXSMM_INLINE
int export_gemm( int argc, char* argv [] ) {
  char* l_a_dt = NULL;
  char* l_b_dt = NULL;
  char* l_comp_dt = NULL;
  char* l_c_dt = NULL;
  libxsmm_datatype l_dtype_a, l_dtype_b, l_dtype_comp, l_dtype_c;
  libxsmm_blasint l_lda = 0, l_ldb = 0, l_ldc = 0;
  libxsmm_blasint l_m = 0, l_n = 0, l_k = 0;
  libxsmm_gemm_prefetch_type l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  libxsmm_generated_code l_generated_code /*= { 0 }*/;
  libxsmm_gemm_shape l_shape;
  libxsmm_gemm_batch_reduce_config l_brconfig;
  libxsmm_gemm_ext_unary_argops l_argops;
  libxsmm_gemm_ext_binary_postops l_postops;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
#if 0
  int l_cfg_flags = 0;
  int l_rls_flags = 0;
#endif
  char l_code_buffer[EXPORT_CODE_BUFFERSIZE];
  const libxsmm_gemm_descriptor* l_xgemm_desc = 0;
  libxsmm_descriptor_blob l_xgemm_blob;
  int l_aligned_a = 0;
  int l_aligned_c = 0;
  int l_trans_a = 0;
  int l_trans_b = 0;
  int l_vnni_a = 0;
  int l_vnni_b = 0;
  int l_vnni_c = 0;
  double l_alpha = 0;
  double l_beta = 0;
  int l_br_count = 1;
  int l_br_type = 0;
  int l_br_unroll = 0;
  size_t l_br_stride_a = 0;
  size_t l_br_stride_b = 0;
  int l_tc_config = 0;
  int l_binary_postop = OP_NONE;
  int l_bop_ld = 0;
  int l_unary_postop = OP_NONE;
  int l_uop_ld = 0;
  int l_is_Ai4Bf16_gemm = 0;
  int l_is_Ai4Bi8_gemm = 0;
  int l_is_Amxfp4Bbf16_gemm = 0;
  int l_is_Amxfp4Bfp32_gemm = 0;
  int l_is_Amxfp4Bi8_gemm = 0;
  int l_is_Abf8Bbf16_gemm = 0;
  int l_is_Abf8Bf16_gemm = 0;
  int l_is_Ahf8Bbf16_gemm = 0;
  int l_is_ext_gemm = 0;
  int l_i = 0;

  /* determine if we have a valid */
  if ( argc == 30 ) {
    l_is_ext_gemm = 0;
  } else if ( argc == 34 ) {
    l_is_ext_gemm = 1;
  } else {
    return EXIT_FAILURE;
  }

  /* print command line */
  fprintf(stdout, "INFO: libxsmm_binaryexport_generator command-line:\nINFO:    ");
  for ( l_i = 0; l_i < 30; l_i++ ) {
    fprintf(stdout, "%s ", argv[l_i]);
  }
  if ( l_is_ext_gemm == 1 ) {
    for ( l_i = 30; l_i < 34; l_i++ ) {
      fprintf(stdout, "%s ", argv[l_i]);
    }
  }
  fprintf(stdout, "\n");

  /* datatypes */
  l_a_dt = argv[4];
  l_b_dt = argv[5];
  l_comp_dt = argv[6];
  l_c_dt = argv[7];

  if (strcmp(argv[4], "I4") == 0) {
    l_is_Ai4Bf16_gemm = 1;
    l_dtype_a         = LIBXSMM_DATATYPE_I8;
  } else if (strcmp(argv[4], "U4") == 0) {
    if (strcmp(argv[5], "F16") == 0) {
      l_is_Ai4Bf16_gemm = 1;
      l_dtype_a         = LIBXSMM_DATATYPE_U8;
    }
    if (strcmp(argv[5], "U8") == 0) {
      l_is_Ai4Bi8_gemm = 1;
      l_dtype_a        = LIBXSMM_DATATYPE_I8;
    }
  } else if (strcmp(argv[4], "MXFP4") == 0) {
    if (strcmp(argv[5], "BF16") == 0) {
      l_is_Amxfp4Bbf16_gemm = 1;
      l_dtype_a             = LIBXSMM_DATATYPE_I8;
    }
    if (strcmp(argv[5], "F32") == 0) {
      l_is_Amxfp4Bfp32_gemm = 1;
      l_dtype_a             = LIBXSMM_DATATYPE_I8;
    }
    if (strcmp(argv[5], "I8") == 0) {
      l_is_Amxfp4Bi8_gemm = 1;
      l_dtype_a           = LIBXSMM_DATATYPE_I8;
    }
  } else if (strcmp(argv[4], "BF8") == 0 && strcmp(argv[5], "BF16") == 0) {
    l_is_Abf8Bbf16_gemm = 1;
    l_dtype_a           = LIBXSMM_DATATYPE_BF16;
  } else if (strcmp(argv[4], "BF8") == 0 && strcmp(argv[5], "F16") == 0 && atoi(argv[20]) > 0) {
    l_is_Abf8Bf16_gemm = 1;
    l_dtype_a          = LIBXSMM_DATATYPE_F16;
  } else if (strcmp(argv[4], "HF8") == 0 && strcmp(argv[5], "BF16") == 0) {
    l_is_Ahf8Bbf16_gemm = 1;
    l_dtype_a           = LIBXSMM_DATATYPE_BF16;
  } else {
    l_dtype_a    = char_to_libxsmm_datatype( l_a_dt );
  }
  l_dtype_b    = char_to_libxsmm_datatype( l_b_dt );
  l_dtype_comp = char_to_libxsmm_datatype( l_comp_dt );
  l_dtype_c    = char_to_libxsmm_datatype( l_c_dt );

  /* GEMM sizes */
  l_m = atoi(argv[8]);
  l_n = atoi(argv[9]);
  l_k = atoi(argv[10]);
  l_lda = atoi(argv[11]);
  l_ldb = atoi(argv[12]);
  l_ldc = atoi(argv[13]);

  /* some sugar */
  l_alpha = atof(argv[14]);
  l_beta = atof(argv[15]);
  l_aligned_a = atoi(argv[16]);
  l_aligned_c = atoi(argv[17]);
  l_trans_a = atoi(argv[18]);
  l_trans_b = atoi(argv[19]);
  l_vnni_a = atoi(argv[20]);
  l_vnni_b = atoi(argv[21]);
  l_vnni_c = atoi(argv[22]);

  /* set value of prefetch flag */
  if (strcmp("nopf", argv[23]) == 0) {
    l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
  }
  else if (strcmp("BL2viaC", argv[23]) == 0) {
    l_prefetch = LIBXSMM_GEMM_PREFETCH_BL2_VIA_C;
  }
  else if (strcmp("curAL2", argv[23]) == 0) {
    l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2_AHEAD;
  }
  else if (strcmp("curAL2_BL2viaC", argv[23]) == 0) {
    l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD;
  }
  else if (strcmp("AL2", argv[23]) == 0) {
    l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2;
  }
  else if (strcmp("AL2_BL2viaC", argv[23]) == 0) {
    l_prefetch = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C;
  } else {
    fprintf(stderr, "ERROR: libxsmm_exportbinary_generator: invalid prefetch option: %s\n", argv[23]);
    return EXIT_FAILURE;
  }

  /* determine BRGEMM config */
  if (strcmp("nobr", argv[24]) == 0) {
    l_br_type = 0;
  }
  else if (strcmp("addrbr", argv[24]) == 0) {
    l_br_type = 1;
  }
  else if (strcmp("offsbr", argv[24]) == 0) {
    l_br_type = 2;
  }
  else if (strcmp("strdbr", argv[24]) == 0) {
    l_br_type = 3;
  }
  else if (strcmp("spmm", argv[24]) == 0) {
    l_br_type = 4;
  } else {
    fprintf(stderr, "ERROR: libxsmm_exportbinary_generator: invalid brgemm option: %s\n", argv[24]);
    return EXIT_FAILURE;
  }
  l_br_count = atoi(argv[25]);
  l_br_unroll = atoi(argv[26]);
  l_br_stride_a = atol(argv[27]);
  l_br_stride_b = atol(argv[28]);

  /* tc hoisting */
  l_tc_config = atoi(argv[29]);

  if ( l_is_ext_gemm > 0 ) {
    l_binary_postop = atoi(argv[30]);
    l_bop_ld = atoi(argv[31]);
    l_unary_postop = atoi(argv[32]);
    l_uop_ld = atoi(argv[33]);
  }

  /* some paramater massaging */
  l_br_count = (l_br_count < 1) ? 1 : l_br_count;
  l_br_count = (l_br_type == 0 || l_br_type == 4) ? 1 : l_br_count;
  l_br_unroll = (l_br_type == 0 || l_br_type == 4) ? 0 : l_br_unroll;

  /* check alpha */
  if ( LIBXSMM_NEQ(l_alpha, 1.0) ) {
    fprintf(stderr, "ERROR: libxsmm_exportbinary_generator: alpha needs to be 1.0!\n");
    return EXIT_FAILURE;
  }

  /* check beta */
  if ( LIBXSMM_NEQ(l_beta, 0.0) && LIBXSMM_NEQ(l_beta, 1.0) ) {
    fprintf(stderr, "ERROR: libxsmm_exportbinary_generator: beta needs to be 0.0 or 1.0!\n");
    return EXIT_FAILURE;
  }

  if ( LIBXSMM_NEQ(l_beta, 0.0) && (l_vnni_c > 0) ) {
    fprintf(stderr, "WARNING: libxsmm_exportbinary_generator: beta needs to be 0.0 when C_vnni fusion is requested... setting beta to 0.0...\n");
    l_beta = 0.0;
  }

  /* check if we have entered supported datatpes */
  if ( !(
         ((l_dtype_a == LIBXSMM_DATATYPE_F64)  && (l_dtype_b == LIBXSMM_DATATYPE_F64)  && (l_dtype_comp == LIBXSMM_DATATYPE_F64) && (l_dtype_c == LIBXSMM_DATATYPE_F64))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_F32)  && (l_dtype_b == LIBXSMM_DATATYPE_F32)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I16)  && (l_dtype_b == LIBXSMM_DATATYPE_I16)  && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_I32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_U8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_I32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_I8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_I32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_I8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_I32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_U8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_I32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_U8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_I8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_I8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_BF16)) ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_I8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_U8)   && (l_dtype_comp == LIBXSMM_DATATYPE_I32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_BF16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_BF16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_BF16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_F32)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF8)   && (l_dtype_b == LIBXSMM_DATATYPE_BF16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_BF16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF8)   && (l_dtype_b == LIBXSMM_DATATYPE_BF16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_HF8)   && (l_dtype_b == LIBXSMM_DATATYPE_BF16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_BF16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_HF8)   && (l_dtype_b == LIBXSMM_DATATYPE_BF16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_BF16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_BF16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_BF16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F16) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_IMPLICIT) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F16) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_IMPLICIT) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F16) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_IMPLICIT) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F16) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_I8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_IMPLICIT) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F16) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_IMPLICIT) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F16) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_U8)   && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_IMPLICIT) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_F16)  && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F16) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_F16)  && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_F16)  && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_IMPLICIT) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_F16)  && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F16) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_F16)  && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F16) && (l_dtype_c == LIBXSMM_DATATYPE_F16))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_F16)  && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_F16)  && (l_dtype_b == LIBXSMM_DATATYPE_F16)  && (l_dtype_comp == LIBXSMM_DATATYPE_IMPLICIT) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF16) && (l_dtype_b == LIBXSMM_DATATYPE_BF16) && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF16) && (l_dtype_b == LIBXSMM_DATATYPE_BF16) && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_BF16)) ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF8)  && (l_dtype_b == LIBXSMM_DATATYPE_BF8)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_BF8)  && (l_dtype_b == LIBXSMM_DATATYPE_BF8)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_BF8))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_HF8)  && (l_dtype_b == LIBXSMM_DATATYPE_HF8)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_F32))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_HF8)  && (l_dtype_b == LIBXSMM_DATATYPE_HF8)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_HF8))  ||
         ((l_dtype_a == LIBXSMM_DATATYPE_HF8)  && (l_dtype_b == LIBXSMM_DATATYPE_HF8)  && (l_dtype_comp == LIBXSMM_DATATYPE_F32) && (l_dtype_c == LIBXSMM_DATATYPE_HF8))
        ) ) {
    fprintf(stderr, "ERROR: libxsmm_exportbinary_generator: Unsupported precion combination: a: %s, b: %s, comp: %s, c: %s!\n", l_a_dt, l_b_dt, l_comp_dt, l_c_dt);
    return EXIT_FAILURE;
  }

  if ((l_dtype_c != LIBXSMM_DATATYPE_BF16) && (l_dtype_c != LIBXSMM_DATATYPE_F16) && (l_dtype_c != LIBXSMM_DATATYPE_BF8) && (l_dtype_c != LIBXSMM_DATATYPE_HF8)) {
    if (l_vnni_c > 0) {
      fprintf(stderr, "ERROR: libxsmm_exportbinary_generator: requested C to be converted to vnni but output prec is not BF16 or BF8 or HF8!\n");
      return EXIT_FAILURE;
    }
  }

  /* set up the flags */
  if ( l_dtype_a == LIBXSMM_DATATYPE_U8 ) {
    l_dtype_a = LIBXSMM_DATATYPE_I8;
    l_flags |= LIBXSMM_GEMM_FLAG_A_UNSIGNED;
  }
  if ( l_dtype_b == LIBXSMM_DATATYPE_U8 ) {
    l_dtype_b = LIBXSMM_DATATYPE_I8;
    l_flags |= LIBXSMM_GEMM_FLAG_B_UNSIGNED;
  }
  if ( (l_dtype_a == LIBXSMM_DATATYPE_I8) && (l_dtype_b == LIBXSMM_DATATYPE_F16) && ((l_dtype_c == LIBXSMM_DATATYPE_F16) || (l_dtype_b == LIBXSMM_DATATYPE_F32)) ) {
    l_flags |= LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF;
  }
  if ( (l_is_Amxfp4Bbf16_gemm == 0) && (l_dtype_a == LIBXSMM_DATATYPE_I8) && (l_dtype_b == LIBXSMM_DATATYPE_BF16) && ((l_dtype_c == LIBXSMM_DATATYPE_BF16) || (l_dtype_c == LIBXSMM_DATATYPE_F32)) ) {
    l_flags |= LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF;
  }
  if (l_is_Ai4Bf16_gemm > 0) {
    l_flags |= LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2;
  }
  if (l_is_Ai4Bf16_gemm > 0) {
    l_flags |= LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT;
  }
  if (l_is_Amxfp4Bi8_gemm > 0) {
    l_flags |= LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI8_INTLV;
  }
  if (l_is_Ai4Bi8_gemm > 0) {
    l_flags |= LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI8_INTLV;
    if (l_br_type == 1 || l_br_type == 2 || l_br_type == 3) {
      l_flags |= LIBXSMM_GEMM_FLAG_USE_MxK_ZPT;
    } else {
      l_flags |= LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT;
    }
  }
  if (l_is_Amxfp4Bbf16_gemm > 0 || l_is_Amxfp4Bfp32_gemm > 0) {
    l_flags |= LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2;
  }
  if (l_is_Amxfp4Bbf16_gemm > 0) {
    l_flags |= LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2;
  }
  if (l_br_type == 4) {
    l_flags |= LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK;
  }
  l_flags |= (0 != l_trans_a ? LIBXSMM_GEMM_FLAG_TRANS_A : 0);
  l_flags |= (0 != l_trans_b ? LIBXSMM_GEMM_FLAG_TRANS_B : 0);
  l_flags |= (0 != l_vnni_a ? LIBXSMM_GEMM_FLAG_VNNI_A : 0);
  l_flags |= (0 != l_vnni_b ? LIBXSMM_GEMM_FLAG_VNNI_B : 0);
  l_flags |= (0 != l_vnni_c ? LIBXSMM_GEMM_FLAG_VNNI_C : 0);
  l_flags |= (0 != l_aligned_a ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0);
  l_flags |= (0 != l_aligned_c ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0);
  l_flags |= (0.0 == l_beta ? LIBXSMM_GEMM_FLAG_BETA_0 : 0);

  /* setting update GEMM struct */
  l_shape = libxsmm_create_gemm_shape( l_m, l_n, l_k, l_lda, l_ldb, l_ldc,
      (l_is_Abf8Bbf16_gemm > 0 || l_is_Abf8Bf16_gemm > 0) ? LIBXSMM_DATATYPE_BF8 : ((l_is_Ahf8Bbf16_gemm > 0) ? LIBXSMM_DATATYPE_HF8 : l_dtype_a), l_dtype_b, l_dtype_c, l_dtype_comp );

  /* setting BRGEMM config struct */
  if (l_br_type == 1) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_ADDRESS;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = (unsigned char)(( l_br_unroll <= 0 ) ? 0 : l_br_count);
  } else if (l_br_type == 2) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = (unsigned char)(( l_br_unroll <= 0 ) ? 0 : l_br_count);
  } else if (l_br_type == 3) {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
    l_brconfig.br_stride_a_hint = l_br_stride_a;
    l_brconfig.br_stride_b_hint = l_br_stride_b;
    l_brconfig.br_unroll_hint = (unsigned char)(( l_br_unroll <= 0 ) ? 0 : l_br_count);
  } else {
    l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;
    l_brconfig.br_stride_a_hint = 0;
    l_brconfig.br_stride_b_hint = 0;
    l_brconfig.br_unroll_hint = 0;
  }

  /* setting ext structs to 0 */
  memset( &l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops) );
  memset( &l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops) );

  /* Setup fusion postops */
  if ( l_is_ext_gemm != 0 ) {
    if (l_binary_postop != OP_NONE ) {
      if (l_binary_postop == COLBIAS_ADD) {
        l_postops.d_in_type      = l_dtype_c;
        l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
        l_postops.d_binary_type  = LIBXSMM_MELTW_TYPE_BINARY_ADD;
        l_postops.ldd            = l_bop_ld;
      }
    }

    if (l_unary_postop != OP_NONE ) {
      if (l_unary_postop == SIGMOID) {
        l_argops.ldcp = l_uop_ld;
        l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_SIGMOID;
      }

      if (l_unary_postop == RELU_NOBITMASK) {
        l_argops.ldcp = l_uop_ld;
        l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
      }

      if (l_unary_postop == RELU_BITMASK) {
        l_argops.ldcp = l_uop_ld;
        l_argops.cp_unary_type  = LIBXSMM_MELTW_TYPE_UNARY_RELU;
        l_argops.cp_unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
      }
    }
  }

  if (l_tc_config != 0) {
#if 0
      l_cfg_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | l_flags;
      l_rls_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | l_flags;
      l_flags |= (LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG);
      cfg_tr.tilecfg = libxsmm_dispatch_tilecfg_gemm( l_shape, l_cfg_flags );
      rls_tr.tilecfg = libxsmm_dispatch_tilecfg_gemm( l_shape, l_rls_flags );
#endif
  }

  if ( l_is_ext_gemm == 0 ) {
    l_xgemm_desc = libxsmm_gemm_descriptor_init_brgemm( &l_xgemm_blob, l_shape,
                                                        l_flags, l_prefetch,
                                                        l_brconfig );
  } else {
    l_xgemm_desc = libxsmm_gemm_descriptor_init_brgemm_ext( &l_xgemm_blob, l_shape,
                                                            l_flags, l_prefetch,
                                                            l_brconfig,
                                                            l_argops, l_postops );
  }

  /* init generated code object */
  l_generated_code.generated_code = l_code_buffer;
  l_generated_code.buffer_size = EXPORT_CODE_BUFFERSIZE;
  l_generated_code.code_size = 0;
  l_generated_code.code_type = 2;
  l_generated_code.last_error = 0;
  l_generated_code.arch = 0;

  /* set arch */
  l_generated_code.arch = libxsmm_cpuid_id( argv[1] );
  if ( l_generated_code.arch == LIBXSMM_TARGET_ARCH_UNKNOWN ) {
    fprintf(stderr, "ERROR: libxsmm_exportbinary_generator: unkown architecture: %s\n", argv[1]);
    return EXIT_FAILURE;
  }

  /* generate the binary code */
  libxsmm_generator_gemm_kernel( &l_generated_code, l_xgemm_desc );

  /* check for errors during code generation */
  if ( l_generated_code.last_error != 0 ) {
    fprintf(stderr, "ERROR: libxsmm_exportbinary_generator: code gen failed with error: %s\n", libxsmm_strerror( l_generated_code.last_error ));
    return EXIT_FAILURE;
  } else {
    fprintf(stdout, "SUCCESS: libxsmm_exportbinary_generator: gemm/gemm_ext was successfully generated. Binary-Size: %i\n", l_generated_code.code_size );
  }

  /* append code to source file */
  {
    FILE *const l_file_handle = fopen( argv[2], "w" );
    if ( l_file_handle != NULL ) {
      fwrite( (const unsigned char*)l_generated_code.generated_code, sizeof(unsigned char), l_generated_code.code_size, l_file_handle );
      fclose( l_file_handle );
      fprintf(stdout, "SUCCESS: libxsmm_exportbinary_generator: gemm/gemm_ext was successfylly written to file.\n");
    } else {
      fprintf(stderr, "ERROR: libxsmm_exportbinary_generator: could not write to into destination source file!\n");
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}

int main(int argc, char* argv []) {
  int ret = EXIT_SUCCESS;

  /* we need at least 4 entries: self arch export-filename kernel */
  if ( argc < 4 ) {
    print_help();
    return EXIT_FAILURE;
  }

  if (strcmp(argv[3], "gemm") == 0) {
    ret = export_gemm( argc, argv );
  } else if (strcmp(argv[3], "gemmext") == 0) {
    ret = export_gemm( argc, argv );
  } else {
    print_help();
    ret = EXIT_FAILURE;
  }

  return ret;
}
