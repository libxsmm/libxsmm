/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_TYPEDEFS_H
#define LIBXSMM_TYPEDEFS_H

#include "libxsmm_macros.h"

/** Check ILP64 configuration for sanity. */
#if !defined(LIBXSMM_ILP64) || (0 == LIBXSMM_ILP64 && defined(MKL_ILP64))
# error "Inconsistent ILP64 configuration detected!"
#elif (0 != LIBXSMM_ILP64 && !defined(MKL_ILP64))
# define MKL_ILP64
#endif
#if (0 != LIBXSMM_ILP64)
# define LIBXSMM_BLASINT_NBITS 64
# define LIBXSMM_BLASINT long long
#else /* LP64 */
# define LIBXSMM_BLASINT_NBITS 32
# define LIBXSMM_BLASINT int
#endif

/** Generic prefetches; similar to LIBXSMM_PREFETCH_AUTO (libxsmm_frontend.h) */
#define LIBXSMM_PREFETCH_SIGONLY 1
#define LIBXSMM_PREFETCH_NONE 0

/** Helper macro for type names. */
#define LIBXSMM_TYPENAME(TYPE) LIBXSMM_STRINGIFY(LIBXSMM_CONCATENATE(LIBXSMM_TYPENAME_, TYPE))
#define LIBXSMM_TYPENAME_double f64
#define LIBXSMM_TYPENAME_float f32
#define LIBXSMM_TYPENAME_libxsmm_bfloat16 bf16
#define LIBXSMM_TYPENAME_int i32
#define LIBXSMM_TYPENAME_short i16
#define LIBXSMM_TYPENAME_char i8

/** Helper macro for type information: INFO := { FP }. */
#define LIBXSMM_TYPEINFO(TYPE, INFO) LIBXSMM_CONCATENATE4(LIBXSMM_TYPEINFO_, INFO, _, TYPE)
#define LIBXSMM_TYPEINFO_FP_double 1
#define LIBXSMM_TYPEINFO_FP_float 1
#define LIBXSMM_TYPEINFO_FP_libxsmm_bfloat16 1
#define LIBXSMM_TYPEINFO_FP_int 0
#define LIBXSMM_TYPEINFO_FP_short 0
#define LIBXSMM_TYPEINFO_FP_char 0

/** Helper macro for type postfixes. */
#define LIBXSMM_TYPESYMBOL(TYPE) LIBXSMM_CONCATENATE(LIBXSMM_TYPESYMBOL_, TYPE)
#define LIBXSMM_TYPESYMBOL_double F64
#define LIBXSMM_TYPESYMBOL_float F32
#define LIBXSMM_TYPESYMBOL_libxsmm_bfloat16 BF16
#define LIBXSMM_TYPESYMBOL_int I32
#define LIBXSMM_TYPESYMBOL_short I16
#define LIBXSMM_TYPESYMBOL_char I8

#define LIBXSMM_TYPESIZE(ENUM) ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_F64  ? 8 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_F32  ? 4 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_BF16 ? 2 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_I32  ? 4 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_I16  ? 2 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_I8   ? 1 : ( \
  0/*invalid*/)))))))

/* Get input or output precision */
#define LIBXSMM_GETENUM_INP(SRC) ((SRC) & 0x0F)
#define LIBXSMM_GETENUM_OUT(SRC) (0 == ((SRC) >> 4) ? LIBXSMM_GETENUM_INP(SRC) : ((SRC) >> 4))
/* Get/Set input and output precision */
#define LIBXSMM_GETENUM(INP, OUT) (((INP) == (OUT)) ? (INP) : ((INP) | ((OUT) << 4)))
#define LIBXSMM_SETENUM(DST, INP, OUT) DST = LIBXSMM_GETENUM(INP, OUT)

/* Construct an enumerator (libxsmm_datatype) from a built-in type (float, double, etc.). */
#define LIBXSMM_DATATYPE(TYPE) LIBXSMM_CONCATENATE(LIBXSMM_DATATYPE_, LIBXSMM_TYPESYMBOL(TYPE))
/* Construct a type-id from built-in input/output types (float, double, etc.). */
#define LIBXSMM_DATATYPE2(ITYPE, OTYPE) LIBXSMM_GETENUM(LIBXSMM_DATATYPE(ITYPE), LIBXSMM_DATATYPE(OTYPE))

/* Construct an enumerator (libxsmm_gemm_precision) from a built-in type (float, double, etc.). */
#define LIBXSMM_GEMM_PRECISION(TYPE) LIBXSMM_CONCATENATE(LIBXSMM_GEMM_PRECISION_, LIBXSMM_TYPESYMBOL(TYPE))
/* Construct GEMM-precision from built-in input/output types (float, double, etc.). */
#define LIBXSMM_GEMM_PRECISION2(ITYPE, OTYPE) (libxsmm_gemm_precision)LIBXSMM_GETENUM( \
  LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(OTYPE))

/** Maximum size available to store a descriptor/blob (GEMM, MCOPY, TRANS, TRSM, TRMM). */
#if !defined(LIBXSMM_DESCRIPTOR_MAXSIZE)
# define LIBXSMM_DESCRIPTOR_MAXSIZE 64
#endif
/** Size of the descriptor considered as unique signature. */
#if !defined(LIBXSMM_DESCRIPTOR_SIGSIZE)
# define LIBXSMM_DESCRIPTOR_SIGSIZE LIBXSMM_DESCRIPTOR_MAXSIZE
#endif


/* Support for Bfloat16 */
typedef unsigned short libxsmm_bfloat16;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_bfloat16_hp {
  libxsmm_bfloat16 i[2];
  float f;
} libxsmm_bfloat16_hp;

#if defined(__cplusplus)
namespace tensorflow { struct bfloat16; }
#endif /*__cplusplus*/

/** Integer type for LAPACK/BLAS (LP64: 32-bit, and ILP64: 64-bit). */
typedef LIBXSMM_BLASINT libxsmm_blasint;

/** Type representing sufficient storage space for a GEMM handle. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_gemm_blob { char data[128]; } libxsmm_gemm_blob;
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_gemm_handle libxsmm_gemm_handle;

/** Type representing sufficient storage space for descriptors (GEMM, TCOPY, MCOPY). */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_descriptor_blob {
  char data[LIBXSMM_DESCRIPTOR_MAXSIZE];
} libxsmm_descriptor_blob;

/** Structure storing arguments of GEMM-like routines. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_gemm_descriptor libxsmm_gemm_descriptor;
/** Structure storing arguments of the matrix-copy routine. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_mcopy_descriptor libxsmm_mcopy_descriptor;
/** Structure storing arguments of the transpose routine. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_trans_descriptor libxsmm_trans_descriptor;
/** Structure storing arguments of packed TRSM. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_trsm_descriptor libxsmm_trsm_descriptor;
/** Structure storing arguments of packed TRMM. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_trmm_descriptor libxsmm_trmm_descriptor;
/** Structure storing arguments of packed GETRF. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_getrf_descriptor libxsmm_getrf_descriptor;
/** Structure storing arguments of packed GEMM. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_pgemm_descriptor libxsmm_pgemm_descriptor;

/** Enumerates element/data types. */
typedef enum libxsmm_datatype {
  LIBXSMM_DATATYPE_F64,
  LIBXSMM_DATATYPE_F32,
  LIBXSMM_DATATYPE_BF16,
  LIBXSMM_DATATYPE_I64,
  LIBXSMM_DATATYPE_I32,
  LIBXSMM_DATATYPE_I16,
  LIBXSMM_DATATYPE_I8,
  LIBXSMM_DATATYPE_UNSUPPORTED
} libxsmm_datatype;

/** Denotes the precision/data type of GEMM. */
typedef enum libxsmm_gemm_precision {
  LIBXSMM_GEMM_PRECISION_F64  = LIBXSMM_DATATYPE_F64,
  LIBXSMM_GEMM_PRECISION_F32  = LIBXSMM_DATATYPE_F32,
  LIBXSMM_GEMM_PRECISION_BF16 = LIBXSMM_DATATYPE_BF16,
  LIBXSMM_GEMM_PRECISION_I32  = LIBXSMM_DATATYPE_I32,
  LIBXSMM_GEMM_PRECISION_I16  = LIBXSMM_DATATYPE_I16,
  LIBXSMM_GEMM_PRECISION_I8   = LIBXSMM_DATATYPE_I8
} libxsmm_gemm_precision;

/** Flag enumeration which can be binary ORed. */
typedef enum libxsmm_gemm_flags {
  LIBXSMM_GEMM_FLAG_NONE = 0,
  /** Transpose matrix A. */
  LIBXSMM_GEMM_FLAG_TRANS_A = 1,
  /** Transpose matrix B. */
  LIBXSMM_GEMM_FLAG_TRANS_B = 2,
  /** Transpose matrix A and B. */
  LIBXSMM_GEMM_FLAG_TRANS_AB = LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B,
#if 0
  /** Alpha=0|1 */
  LIBXSMM_GEMM_FLAG_ALPHA_0 = 4,
  /** Alpha=neg|pos */
  LIBXSMM_GEMM_FLAG_ALPHA_S = 8,
#endif
  /** Beta=0|1 */
  LIBXSMM_GEMM_FLAG_BETA_0 = 16,
#if 0
  /** Beta=neg|pos */
  LIBXSMM_GEMM_FLAG_BETA_S = 32,
#endif
  /** Generate aligned load instructions. */
  LIBXSMM_GEMM_FLAG_ALIGN_A = 64,
  /** Aligned load/store instructions. */
  LIBXSMM_GEMM_FLAG_ALIGN_C = 128,
  /** Batch-reduce Ai * Bi. */
  LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS = 256,
  /** Batch-reduce Ai * Bi. */
  LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET = 512,
  /** Batch-reduce Ai * Bi. */
  LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE = 1024,
  /** Aligned C matrix, but using NTS Hint when storing */
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT = 2176,
  /* in case of integer GEMM, if A is unsigned */
  LIBXSMM_GEMM_FLAG_A_UNSIGNED = 4096,
  /* in case of integer GEMM, if B is unsigned */
  LIBXSMM_GEMM_FLAG_B_UNSIGNED = 8192,
  /* in case of integer GEMM, if C is unsigned */
  LIBXSMM_GEMM_FLAG_C_UNSIGNED = 16384,
  /* in case of integer GEMM, if A and B are unsigned */
  LIBXSMM_GEMM_FLAG_AB_UNSIGNED = LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_B_UNSIGNED,
  /* for low precision we also require up-front packed formats "VNNI" for best performance, this flag indicates A */
  LIBXSMM_GEMM_FLAG_VNNI_A = 32768,
  /* for low precision we also require up-front packed formats "VNNI" for best performance, this flag indicates B */
  LIBXSMM_GEMM_FLAG_VNNI_B = 65536,
  /* combined types */
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0                      = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_ADDRESS        = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_ADDRESS = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_OFFSET         = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_OFFSET  = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_STRIDE         = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_STRIDE  = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_A_UNSIGNED                      = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_A_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_ADDRESS_A_UNSIGNED        = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_A_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_ADDRESS_A_UNSIGNED = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS | LIBXSMM_GEMM_FLAG_A_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_OFFSET_A_UNSIGNED         = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_A_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_OFFSET_A_UNSIGNED  = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET | LIBXSMM_GEMM_FLAG_A_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_STRIDE_A_UNSIGNED         = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_A_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_STRIDE_A_UNSIGNED  = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE | LIBXSMM_GEMM_FLAG_A_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_B_UNSIGNED                      = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_B_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_ADDRESS_B_UNSIGNED        = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_B_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_ADDRESS_B_UNSIGNED = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS | LIBXSMM_GEMM_FLAG_B_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_OFFSET_B_UNSIGNED         = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_B_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_OFFSET_B_UNSIGNED  = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET | LIBXSMM_GEMM_FLAG_B_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_STRIDE_B_UNSIGNED         = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_B_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_STRIDE_B_UNSIGNED  = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE | LIBXSMM_GEMM_FLAG_B_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_AB_UNSIGNED                      = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_AB_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_ADDRESS_AB_UNSIGNED        = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_AB_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_ADDRESS_AB_UNSIGNED = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS | LIBXSMM_GEMM_FLAG_AB_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_OFFSET_AB_UNSIGNED         = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_AB_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_OFFSET_AB_UNSIGNED  = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET | LIBXSMM_GEMM_FLAG_AB_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BATCH_REDUCE_STRIDE_AB_UNSIGNED         = LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_AB_UNSIGNED,
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0_BATCH_REDUCE_STRIDE_AB_UNSIGNED  = LIBXSMM_GEMM_FLAG_BETA_0       | LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT | LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE | LIBXSMM_GEMM_FLAG_AB_UNSIGNED,
  /** Marker flag; do not use. */
  LIBXSMM_GEMM_FLAG_INVALID = 131072
} libxsmm_gemm_flags;

/** Flag enumeration which can be binary ORed. */
typedef enum libxsmm_gemm_handle_flags {
  LIBXSMM_GEMM_HANDLE_FLAG_AUTO   = 0,
  LIBXSMM_GEMM_HANDLE_FLAG_COPY_A = 1,
  LIBXSMM_GEMM_HANDLE_FLAG_COPY_B = 2,
  LIBXSMM_GEMM_HANDLE_FLAG_COPY_C = 4
} libxsmm_gemm_handle_flags;

/** Auto-batch flags (can be ORed) applicable to mmbatch_begin/mmbatch_end. */
typedef enum libxsmm_mmbatch_flags {
  /** Handle recorded batch unsynchronized-parallel. */
  LIBXSMM_MMBATCH_FLAG_DEFAULT      = LIBXSMM_GEMM_FLAG_INVALID * 0,
  /** Synchronize among C matrices. */
  LIBXSMM_MMBATCH_FLAG_SYNCHRONIZED = LIBXSMM_GEMM_FLAG_INVALID * 1,
  /** Handle recorded batch sequentially. */
  LIBXSMM_MMBATCH_FLAG_SEQUENTIAL   = LIBXSMM_GEMM_FLAG_INVALID * 2,
  /** Only record a statistic of potential SMMs. */
  LIBXSMM_MMBATCH_FLAG_STATISTIC    = LIBXSMM_GEMM_FLAG_INVALID * 4
} libxsmm_mmbatch_flags;

/** Enumeration of the available prefetch strategies. */
typedef enum libxsmm_gemm_prefetch_type {
  /** No prefetching and no prefetch fn. signature. */
  LIBXSMM_GEMM_PREFETCH_NONE               = LIBXSMM_PREFETCH_NONE,
  /** Only function prefetch signature. */
  LIBXSMM_GEMM_PREFETCH_SIGONLY            = LIBXSMM_PREFETCH_SIGONLY,
  /** Prefetch PA using accesses to A. */
  LIBXSMM_GEMM_PREFETCH_AL2                = 2,
  /** Prefetch PA (aggressive). */
  LIBXSMM_GEMM_PREFETCH_BL2_VIA_C          = 4,
  /** Prefetch A ahead. */
  LIBXSMM_GEMM_PREFETCH_AL2_AHEAD          = 8,
  LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C       = LIBXSMM_GEMM_PREFETCH_BL2_VIA_C | LIBXSMM_GEMM_PREFETCH_AL2,
  LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD = LIBXSMM_GEMM_PREFETCH_BL2_VIA_C | LIBXSMM_GEMM_PREFETCH_AL2_AHEAD,
  /** Backward compatibility: AL2CL2BL2_VIA_C is an alias for AL2BL2_VIA_C (Eigen library). */
  LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C         = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C,
  /* current B into L1 */
  LIBXSMM_GEMM_PREFETCH_BL1                = 16
} libxsmm_gemm_prefetch_type;

/** Flag enumeration which can be binary ORed. */
typedef enum libxsmm_matcopy_flags {
  LIBXSMM_MATCOPY_FLAG_DEFAULT = 0,
  /** If set, then use zero matrix as source */
  LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE = 1
} libxsmm_matcopy_flags;

typedef enum libxsmm_dnn_tensor_format {
  /* use LIBXSMM internal format, we need to copy data into that */
  LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM  = 1,
  /* use NHWC format internally, this allows no-copy operations */
  LIBXSMM_DNN_TENSOR_FORMAT_NHWC     = 2,
  /* use NCHW format internally, this will include shadow copies, not preferred */
  LIBXSMM_DNN_TENSOR_FORMAT_NCHW     = 4,
  /* use RSCK format internally, this allows no-copy operations */
  LIBXSMM_DNN_TENSOR_FORMAT_RSCK     = 8,
  /* use KCRS format internally, this will include shadow copies, not preferred */
  LIBXSMM_DNN_TENSOR_FORMAT_KCRS     = 16,
  LIBXSMM_DNN_TENSOR_FORMAT_CK       = 32,
  LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED = 64,
  LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED = 128,
  LIBXSMM_DNN_TENSOR_FORMAT_NC       = 256
} libxsmm_dnn_tensor_format;

/** Denotes the element/pixel type of an image/channel. */
typedef enum libxsmm_dnn_datatype {
  LIBXSMM_DNN_DATATYPE_F64  = LIBXSMM_DATATYPE_F64,
  LIBXSMM_DNN_DATATYPE_F32  = LIBXSMM_DATATYPE_F32,
  LIBXSMM_DNN_DATATYPE_BF16 = LIBXSMM_DATATYPE_BF16,
  LIBXSMM_DNN_DATATYPE_I32  = LIBXSMM_DATATYPE_I32,
  LIBXSMM_DNN_DATATYPE_I16  = LIBXSMM_DATATYPE_I16,
  LIBXSMM_DNN_DATATYPE_I8   = LIBXSMM_DATATYPE_I8
} libxsmm_dnn_datatype;

typedef enum libxsmm_dnn_conv_option {
  /* we get default settings */
  LIBXSMM_DNN_CONV_OPTION_NONE = 0,
  /* overwrite results buffer (set it to zero before running the operations) */
  LIBXSMM_DNN_CONV_OPTION_OVERWRITE = 1,
  /* activations are stored unsigned */
  /* @TODO check if we still need this option */
  LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED = 2,
  /* reduce filters externally to op */
  LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE = 4,
  /* external filter transpose to bwd convolutions */
  LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE = 8,
  /* external filter transpose to bwd convolutions */
  LIBXSMM_DNN_CONV_OPTION_UPD_NO_INPUT_TRANSPOSE = 16,
  /* Down-convert for BF16 using RNE rounding */
  LIBXSMM_DNN_CONV_OPTION_F32_BF16_CVT_RNE = 32,
  /* compound types */
  LIBXSMM_DNN_CONV_OPTION_F32_BF16_CVT_RNE_OVERWRITE = LIBXSMM_DNN_CONV_OPTION_OVERWRITE | LIBXSMM_DNN_CONV_OPTION_F32_BF16_CVT_RNE,
  LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED_OVERWRITE = LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED | LIBXSMM_DNN_CONV_OPTION_OVERWRITE,
  LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE_OVERWRITE = LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE | LIBXSMM_DNN_CONV_OPTION_OVERWRITE,
  LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE_OVERWRITE = LIBXSMM_DNN_CONV_OPTION_OVERWRITE | LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE,
  LIBXSMM_DNN_CONV_OPTION_UPD_NO_INPUT_TRANSPOSE_OVERWRITE = LIBXSMM_DNN_CONV_OPTION_OVERWRITE | LIBXSMM_DNN_CONV_OPTION_UPD_NO_INPUT_TRANSPOSE,
  LIBXSMM_DNN_CONV_OPTION_NO_TRANSPOSES_OVERWRITE = LIBXSMM_DNN_CONV_OPTION_OVERWRITE | LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE | LIBXSMM_DNN_CONV_OPTION_UPD_NO_INPUT_TRANSPOSE
} libxsmm_dnn_conv_option;

typedef enum libxsmm_dnn_fusedbatchnorm_fuse_order {
  /* the fuse order is: 1. BN, 2. element-wise 3. RELU */
  LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU = 0
} libxsmm_dnn_fusedbatchnorm_fuse_order;

typedef enum libxsmm_dnn_fusedbatchnorm_fuse_op {
  /* the fuse order is: 1. BN, 2. element-wise 3. RELU */
  LIBXSMM_DNN_FUSEDBN_OPS_BN = 1,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE = 2,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS = 4,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED = 8,
  LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE = 16,
  LIBXSMM_DNN_FUSEDBN_OPS_RELU = 32,
  LIBXSMM_DNN_FUSEDBN_OPS_RELU_WITH_MASK = 64,
  LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE_RELU = LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU_WITH_MASK,
  LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE = LIBXSMM_DNN_FUSEDBN_OPS_BN | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE,
  LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BN | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDBN_OPS_BN | LIBXSMM_DNN_FUSEDBN_OPS_RELU_WITH_MASK,
  LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BN | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDBN_OPS_BN | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU_WITH_MASK,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE | LIBXSMM_DNN_FUSEDBN_OPS_RELU_WITH_MASK,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU_WITH_MASK,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_ELTWISE = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS | LIBXSMM_DNN_FUSEDBN_OPS_RELU_WITH_MASK,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_ELTWISE_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_ELTWISE_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU_WITH_MASK,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED_ELTWISE = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED | LIBXSMM_DNN_FUSEDBN_OPS_RELU_WITH_MASK,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED_ELTWISE_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED_ELTWISE_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU_WITH_MASK
} libxsmm_dnn_fusedbatchnorm_fuse_op;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_fusedbatchnorm_desc {
  int partN;                                 /* number of images in mini-batch, used for all elementwise computations */
  int fullN;                                 /* number of images in mini-batch, used for statistics computations */
  int C;                                     /* number of input feature maps */
  int H;                                     /* height of input image */
  int W;                                     /* width of input image */
  int u;                                     /* vertical stride */
  int v;                                     /* horizontal stride */
  int pad_h_in;                              /* height of physical zero-padding in input buffer */
  int pad_w_in;                              /* width of physical zero-padding in input buffer */
  int pad_h_out;                             /* height of physical zero-padding in output buffer */
  int pad_w_out;                             /* width of physical zero-padding in output buffer */
  int threads;                               /* number of threads used */
  libxsmm_dnn_datatype datatype_in;          /* datatype used for all input related buffers */
  libxsmm_dnn_datatype datatype_out;         /* datatype used for all output related buffers */
  libxsmm_dnn_datatype datatype_stats;       /* datatype used for all stats related buffers */
  libxsmm_dnn_tensor_format buffer_format;   /* format which is for activation buffers */
  libxsmm_dnn_fusedbatchnorm_fuse_order fuse_order; /* additional options */
  libxsmm_dnn_fusedbatchnorm_fuse_op fuse_ops;      /* used ops into convolutions */
} libxsmm_dnn_fusedbatchnorm_desc;

typedef enum libxsmm_dnn_fusedgroupnorm_fuse_order {
  /* the fuse order is: 1. BN, 2. element-wise 3. RELU */
  LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU = 0
} libxsmm_dnn_fusedgroupnorm_fuse_order;

typedef enum libxsmm_dnn_fusedgroupnorm_fuse_op {
  /* the fuse order is: 1. GN, 2. element-wise 3. RELU */
  LIBXSMM_DNN_FUSEDGN_OPS_GN = 1,
  LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE = 2,
  LIBXSMM_DNN_FUSEDGN_OPS_RELU = 4,
  LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK = 8,
  LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU = LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDGN_OPS_RELU,
  LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK,
  LIBXSMM_DNN_FUSEDGN_OPS_GN_ELTWISE = LIBXSMM_DNN_FUSEDGN_OPS_GN | LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE,
  LIBXSMM_DNN_FUSEDGN_OPS_GN_RELU = LIBXSMM_DNN_FUSEDGN_OPS_GN | LIBXSMM_DNN_FUSEDGN_OPS_RELU,
  LIBXSMM_DNN_FUSEDGN_OPS_GN_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDGN_OPS_GN | LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK,
  LIBXSMM_DNN_FUSEDGN_OPS_GN_ELTWISE_RELU = LIBXSMM_DNN_FUSEDGN_OPS_GN | LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDGN_OPS_RELU,
  LIBXSMM_DNN_FUSEDGN_OPS_GN_ELTWISE_RELU_WITH_MASK = LIBXSMM_DNN_FUSEDGN_OPS_GN | LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK
} libxsmm_dnn_fusedgroupnorm_fuse_op;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_fusedgroupnorm_desc {
  int N;                                     /* number of images in mini-batch */
  int G;                                     /* groups of channels to norm */
  int C;                                     /* number of input feature maps */
  int H;                                     /* height of input image */
  int W;                                     /* width of input image */
  int u;                                     /* vertical stride */
  int v;                                     /* horizontal stride */
  int pad_h_in;                              /* height of physical zero-padding in input buffer */
  int pad_w_in;                              /* width of physical zero-padding in input buffer */
  int pad_h_out;                             /* height of physical zero-padding in output buffer */
  int pad_w_out;                             /* width of physical zero-padding in output buffer */
  int threads;                               /* number of threads used */
  libxsmm_dnn_datatype datatype_in;          /* datatype used for all input related buffers */
  libxsmm_dnn_datatype datatype_out;         /* datatype used for all output related buffers */
  libxsmm_dnn_datatype datatype_stats;       /* datatype used for all stats related buffers */
  libxsmm_dnn_tensor_format buffer_format;   /* format which is for activation buffers */
  libxsmm_dnn_fusedgroupnorm_fuse_order fuse_order; /* additional options */
  libxsmm_dnn_fusedgroupnorm_fuse_op fuse_ops;      /* used ops into convolutions */
} libxsmm_dnn_fusedgroupnorm_desc;

/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (double-precision). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_dmmfunction)(const double* a, const double* b, double* c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (single-precision). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_smmfunction)(const float* a, const float* b, float* c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (bf16, fp32-accumulate). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_bsmmfunction)(const libxsmm_bfloat16* a, const libxsmm_bfloat16* b, float* c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (bf16, fp32-accumulate). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_bmmfunction)(const libxsmm_bfloat16* a, const libxsmm_bfloat16* b, libxsmm_bfloat16* c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (low-precision). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_wimmfunction)(const short* a, const short* b, int* c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (int8, int32 accumulate). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_ssbimmfunction)(const          char* a, const          char* b, int* c, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_usbimmfunction)(const unsigned char* a, const          char* b, int* c, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_subimmfunction)(const          char* a, const unsigned char* b, int* c, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_uubimmfunction)(const unsigned char* a, const unsigned char* b, int* c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (int8, int32 accumulate, int8 downconvert). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_sububmmfunction)(const          char* a, const unsigned char* b, unsigned char* c, float* scf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_dmmfunction_reducebatch_addr)(const double** a, const double** b, double* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_smmfunction_reducebatch_addr)(const float** a, const float** b, float* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_bsmmfunction_reducebatch_addr)(const libxsmm_bfloat16** a, const libxsmm_bfloat16** b, float* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_bmmfunction_reducebatch_addr)(const libxsmm_bfloat16** a, const libxsmm_bfloat16** b, libxsmm_bfloat16* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_wimmfunction_reducebatch_addr)(const short** a, const short** b, int* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_ssbimmfunction_reducebatch_addr)(const          char** a, const          char** b, int* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_usbimmfunction_reducebatch_addr)(const unsigned char** a, const          char** b, int* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_subimmfunction_reducebatch_addr)(const          char** a, const unsigned char** b, int* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_uubimmfunction_reducebatch_addr)(const unsigned char** a, const unsigned char** b, int* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_sububmmfunction_reducebatch_addr)(const          char** a, const unsigned char** b, unsigned char* c, const unsigned long long* count, float* scf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_dmmfunction_reducebatch_offs)(const double* a, const double* b, double* c, const unsigned long long* count, const unsigned long long* a_offs, const unsigned long long* b_offs, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_smmfunction_reducebatch_offs)(const float* a, const float* b, float* c, const unsigned long long* count, const unsigned long long* a_offs, const unsigned long long* b_offs, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_bsmmfunction_reducebatch_offs)(const libxsmm_bfloat16* a, const libxsmm_bfloat16* b, float* c, const unsigned long long* count, const unsigned long long* a_offs, const unsigned long long* b_offs, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_bmmfunction_reducebatch_offs)(const libxsmm_bfloat16* a, const libxsmm_bfloat16* b, libxsmm_bfloat16* c, const unsigned long long* count, const unsigned long long* a_offs, const unsigned long long* b_offs, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_wimmfunction_reducebatch_offs)(const short* a, const short* b, int* c, const unsigned long long* count, const unsigned long long* a_offs, const unsigned long long* b_offs, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_ssbimmfunction_reducebatch_offs)(const          char* a, const          char* b, int* c, const unsigned long long* count, const unsigned long long* a_offs, const unsigned long long* b_offs, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_usbimmfunction_reducebatch_offs)(const unsigned char* a, const          char* b, int* c, const unsigned long long* count, const unsigned long long* a_offs, const unsigned long long* b_offs, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_subimmfunction_reducebatch_offs)(const          char* a, const unsigned char* b, int* c, const unsigned long long* count, const unsigned long long* a_offs, const unsigned long long* b_offs, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_uubimmfunction_reducebatch_offs)(const unsigned char* a, const unsigned char* b, int* c, const unsigned long long* count, const unsigned long long* a_offs, const unsigned long long* b_offs, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_sububmmfunction_reducebatch_offs)(const          char* a, const unsigned char* b, unsigned char* c, const unsigned long long* count, const unsigned long long* a_offs, const unsigned long long* b_offs, float* scf, ...);

LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_dmmfunction_reducebatch_strd)(const double* a, const double* b, double* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_smmfunction_reducebatch_strd)(const float* a, const float* b, float* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_bsmmfunction_reducebatch_strd)(const libxsmm_bfloat16* a, const libxsmm_bfloat16* b, float* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_bmmfunction_reducebatch_strd)(const libxsmm_bfloat16* a, const libxsmm_bfloat16* b, libxsmm_bfloat16* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_wimmfunction_reducebatch_strd)(const short* a, const short* b, int* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_ssbimmfunction_reducebatch_strd)(const          char* a, const          char* b, int* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_usbimmfunction_reducebatch_strd)(const unsigned char* a, const          char* b, int* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_subimmfunction_reducebatch_strd)(const          char* a, const unsigned char* b, int* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_uubimmfunction_reducebatch_strd)(const unsigned char* a, const unsigned char* b, int* c, const unsigned long long* count, ...);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_sububmmfunction_reducebatch_strd)(const          char* a, const unsigned char* b, unsigned char* c, const unsigned long long* count, float* scf, ...);

/** Function type which is either libxsmm_smmfunction or libxsmm_dmmfunction (weak-typed). */
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_xmmfunction {
  void (*xmm)(const void* a, const void* b, void* c, ...);
  void (*xbm)(const void** a, const void** b, void* c, const unsigned long long* count, ...);
  libxsmm_dmmfunction dmm; libxsmm_smmfunction smm; libxsmm_wimmfunction wimm; libxsmm_bsmmfunction bsmm; libxsmm_bmmfunction bmm;
  libxsmm_ssbimmfunction ssbimm; libxsmm_usbimmfunction usbimm; libxsmm_subimmfunction subimm; libxsmm_uubimmfunction uubimm; libxsmm_sububmmfunction sububmm;
  libxsmm_dmmfunction_reducebatch_addr dmra; libxsmm_smmfunction_reducebatch_addr smra; libxsmm_bsmmfunction_reducebatch_addr bsmra; libxsmm_bmmfunction_reducebatch_addr bmra;
  libxsmm_wimmfunction_reducebatch_addr wimra; libxsmm_ssbimmfunction_reducebatch_addr ssbimra; libxsmm_usbimmfunction_reducebatch_addr usbimra; libxsmm_subimmfunction_reducebatch_addr subimra; libxsmm_uubimmfunction_reducebatch_addr uubimra;
  libxsmm_sububmmfunction_reducebatch_addr sububmra;
  libxsmm_dmmfunction_reducebatch_offs dmro; libxsmm_smmfunction_reducebatch_offs smro; libxsmm_bsmmfunction_reducebatch_offs bsmro; libxsmm_bmmfunction_reducebatch_offs bmro;
  libxsmm_wimmfunction_reducebatch_offs wimro; libxsmm_ssbimmfunction_reducebatch_offs ssbimro; libxsmm_usbimmfunction_reducebatch_offs usbimro; libxsmm_subimmfunction_reducebatch_offs subimro; libxsmm_uubimmfunction_reducebatch_offs uubimro;
  libxsmm_sububmmfunction_reducebatch_offs sububmro;
  libxsmm_dmmfunction_reducebatch_strd dmrs; libxsmm_smmfunction_reducebatch_strd smrs; libxsmm_bsmmfunction_reducebatch_strd bsmrs; libxsmm_bmmfunction_reducebatch_strd bmrs;
  libxsmm_wimmfunction_reducebatch_strd wimrs; libxsmm_ssbimmfunction_reducebatch_strd ssbimrs; libxsmm_usbimmfunction_reducebatch_strd usbimrs; libxsmm_subimmfunction_reducebatch_strd subimrs; libxsmm_uubimmfunction_reducebatch_strd uubimrs;
  libxsmm_sububmmfunction_reducebatch_strd sububmrs;
} libxsmm_xmmfunction;

/** Determines the kernel kind. */
typedef enum libxsmm_kernel_kind {
  /** Matrix multiplication kernel */
  LIBXSMM_KERNEL_KIND_MATMUL  = 0,
  /** Matcopy kernel kind */
  LIBXSMM_KERNEL_KIND_MCOPY   = 1,
  /** Transpose kernel kind */
  LIBXSMM_KERNEL_KIND_TRANS   = 2,
  /** GEMM/packed kernel kind */
  LIBXSMM_KERNEL_KIND_PGEMM   = 3,
  /** GEMM/packed kernel kind */
  LIBXSMM_KERNEL_KIND_GETRF   = 4,
  /** TRMM kernel kind */
  LIBXSMM_KERNEL_KIND_TRMM    = 5,
  /** TRSM kernel kind */
  LIBXSMM_KERNEL_KIND_TRSM    = 6,
  /** User-defined kernels */
  LIBXSMM_KERNEL_KIND_USER    = 7,
  /** Not a JIT kernel */
  LIBXSMM_KERNEL_UNREGISTERED = 8
} libxsmm_kernel_kind;

/** Specialized function for matrix-copy (weak-typed). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_xmcopyfunction)(
  const void* in, const unsigned int* ldi, void* out, const unsigned int* ldo, ...);

/** Specialized function for transpose (weak-typed). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_xtransfunction)(
  const void* in, const unsigned int* ldi, void* out, const unsigned int* ldo);

/** Specialized function for packed GEMM (weak-typed). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_pgemm_xfunction)(
  const void* a, const void* b, void* c);

/** Specialized function for packed GEMM (weak-typed). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_getrf_xfunction)(
  const void* a, const void* b, void* c);

/** Specialized function for TRMM (weak-typed). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_trmm_xfunction)(
  const void* a, const void* b, void* c);

/** Specialized function for TRSM (weak-typed). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_trsm_xfunction)(
  const void* a, const void* b, void* c);

/** Structure to receive information about GEMM-kernels (libxsmm_get_mmkernel_info). */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_mmkernel_info {
  /** Input/output data-type */
  libxsmm_gemm_precision iprecision, oprecision;
  /** Prefetch strategy. */
  libxsmm_gemm_prefetch_type prefetch;
  /** Leading dimensions. */
  unsigned int lda, ldb, ldc;
  /** Extents/shape. */
  unsigned int m, n, k;
  /** Set of flags. */
  int flags;
} libxsmm_mmkernel_info;

/** Structure to receive information about transpose-kernels (libxsmm_get_transkernel_info). */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_transkernel_info {
  /** LD, M, and N. */
  unsigned int ldo, m, n;
  /** Size of data element. */
  unsigned int typesize;
} libxsmm_transkernel_info;

/** Structure to receive information about matrix-copy kernels (libxsmm_get_mcopykernel_info). */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_mcopykernel_info {
  /** LDx, M, and N. */
  unsigned int ldi, ldo, m, n;
  /** Size of data element. */
  unsigned int typesize;
  /** Boolean value. */
  int prefetch;
  /** Set of flags. */
  int flags;
} libxsmm_mcopykernel_info;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_kernel_info {
  libxsmm_kernel_kind kind;
  /** Number of FLoating Point OperationS (FLOPS). */
  unsigned int nflops;
  /** Code size (Bytes). */
  size_t code_size;
} libxsmm_kernel_info;

/** Structure to receive information about the code registry status (libxsmm_get_registry_info). */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_registry_info {
  size_t capacity, size, nbytes, nstatic, ncache;
} libxsmm_registry_info;

#endif /*LIBXSMM_TYPEDEFS_H*/

