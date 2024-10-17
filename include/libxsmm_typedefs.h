/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
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

/** Generic prefetch applicable for all domains. */
#define LIBXSMM_PREFETCH_NONE 0
/** Attempt to automatically select a strategy. */
#define LIBXSMM_PREFETCH_AUTO -1

/** Helper macro for type names. */
#define LIBXSMM_TYPENAME(TYPE) LIBXSMM_STRINGIFY(LIBXSMM_CONCATENATE(LIBXSMM_TYPENAME_, TYPE))
#define LIBXSMM_TYPENAME_double f64
#define LIBXSMM_TYPENAME_float f32
#define LIBXSMM_TYPENAME_libxsmm_bfloat16 bf16
#define LIBXSMM_TYPENAME_libxsmm_float16 f16
#define LIBXSMM_TYTPNAME_libxsmm_bfloat8 bf8
#define LIBXSMM_TYTPNAME_libxsmm_hfloat8 hf8
#define LIBXSMM_TYPENAME_int i32
#define LIBXSMM_TYPENAME_short i16
#define LIBXSMM_TYPENAME_char i8

/** Helper macro for type information: INFO := { FP }. */
#define LIBXSMM_TYPEINFO(TYPE, INFO) LIBXSMM_CONCATENATE4(LIBXSMM_TYPEINFO_, INFO, _, TYPE)
#define LIBXSMM_TYPEINFO_FP_double 1
#define LIBXSMM_TYPEINFO_FP_float 1
#define LIBXSMM_TYPEINFO_FP_libxsmm_bfloat16 1
#define LIBXSMM_TYPEINFO_FP_libxsmm_float16 1
#define LIBXSMM_TYPEINFO_FP_libxsmm_bfloat8 1
#define LIBXSMM_TYPEINFO_FP_libxsmm_hfloat8 1
#define LIBXSMM_TYPEINFO_FP_int 0
#define LIBXSMM_TYPEINFO_FP_short 0
#define LIBXSMM_TYPEINFO_FP_char 0

/** Helper macro for type postfixes. */
#define LIBXSMM_TYPESYMBOL(TYPE) LIBXSMM_CONCATENATE(LIBXSMM_TYPESYMBOL_, TYPE)
#define LIBXSMM_TYPESYMBOL_double F64
#define LIBXSMM_TYPESYMBOL_float F32
#define LIBXSMM_TYPESYMBOL_libxsmm_bfloat16 BF16
#define LIBXSMM_TYPESYMBOL_libxsmm_float16 F16
#define LIBXSMM_TYPESYMBOL_libxsmm_bfloat8 BF8
#define LIBXSMM_TYPESYMBOL_libxsmm_hfloat8 HF8
#define LIBXSMM_TYPESYMBOL_int I32
#define LIBXSMM_TYPESYMBOL_short I16
#define LIBXSMM_TYPESYMBOL_char I8

#define LIBXSMM_TYPESIZE(ENUM) ( \
  (LIBXSMM_DATATYPE_F64  == ((int)(ENUM))) ? 8 : ( \
  (LIBXSMM_DATATYPE_F32  == ((int)(ENUM))) ? 4 : ( \
  (LIBXSMM_DATATYPE_BF16 == ((int)(ENUM))) ? 2 : ( \
  (LIBXSMM_DATATYPE_F16  == ((int)(ENUM))) ? 2 : ( \
  (LIBXSMM_DATATYPE_BF8  == ((int)(ENUM))) ? 1 : ( \
  (LIBXSMM_DATATYPE_HF8  == ((int)(ENUM))) ? 1 : ( \
  (LIBXSMM_DATATYPE_I64  == ((int)(ENUM))) ? 8 : ( \
  (LIBXSMM_DATATYPE_U64  == ((int)(ENUM))) ? 8 : ( \
  (LIBXSMM_DATATYPE_I32  == ((int)(ENUM))) ? 4 : ( \
  (LIBXSMM_DATATYPE_U32  == ((int)(ENUM))) ? 4 : ( \
  (LIBXSMM_DATATYPE_I16  == ((int)(ENUM))) ? 2 : ( \
  (LIBXSMM_DATATYPE_U16  == ((int)(ENUM))) ? 2 : ( \
  (LIBXSMM_DATATYPE_I8   == ((int)(ENUM))) ? 1 : ( \
  (LIBXSMM_DATATYPE_U8   == ((int)(ENUM))) ? 1 : ( \
  (LIBXSMM_ASSERT_MSG(0/*false*/, "Invalid datatype"), \
    0/*invalid*/))))))))))))))))

/* Get input precision datatype (preserves unsigned datatype) */
#define LIBXSMM_GETENUM_UNP(SRC) ((SRC) & 0x0F)
/* Get signed precision datatype regardless of signed or unsigned input */
#define LIBXSMM_GETENUM_INP(SRC) ( \
  (LIBXSMM_DATATYPE_F64         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_F64 : ( \
  (LIBXSMM_DATATYPE_F32         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_F32 : ( \
  (LIBXSMM_DATATYPE_BF16        == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_BF16 : ( \
  (LIBXSMM_DATATYPE_F16         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_F16 : ( \
  (LIBXSMM_DATATYPE_BF8         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_BF8 : ( \
  (LIBXSMM_DATATYPE_HF8         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_HF8 : ( \
  (LIBXSMM_DATATYPE_I64         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_I64 : ( \
  (LIBXSMM_DATATYPE_U64         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_I64 : ( \
  (LIBXSMM_DATATYPE_I32         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_I32 : ( \
  (LIBXSMM_DATATYPE_U32         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_I32 : ( \
  (LIBXSMM_DATATYPE_I16         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_I16 : ( \
  (LIBXSMM_DATATYPE_U16         == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_I16 : ( \
  (LIBXSMM_DATATYPE_I8          == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_I8 : ( \
  (LIBXSMM_DATATYPE_U8          == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_I8 : ( \
  (LIBXSMM_DATATYPE_IMPLICIT    == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_IMPLICIT : ( \
  (LIBXSMM_DATATYPE_UNSUPPORTED == LIBXSMM_GETENUM_UNP(SRC)) ? LIBXSMM_DATATYPE_UNSUPPORTED : ( \
  (LIBXSMM_ASSERT_MSG(0/*false*/, "Invalid datatype"), \
    0/*invalid*/))))))))))))))))))

/* Get output precision datatype (preserves unsigned datatype) */
#define LIBXSMM_GETENUM_UOT(SRC) (0 == ((SRC) >> 4) ? LIBXSMM_GETENUM_UNP(SRC) : ((SRC) >> 4))
/* Get signed precision datatype regardless of signed or unsigned output */
#define LIBXSMM_GETENUM_OUT(SRC) (0 == ((SRC) >> 4) ? LIBXSMM_GETENUM_INP(SRC) : ((SRC) >> 4))
/* Get/Set input and output precision */
#define LIBXSMM_GETENUM(INP, OUT) (((INP) == (OUT)) \
  ? ((unsigned char)((INP))) \
  : ((unsigned char)((INP) | ((unsigned char)((OUT) << 4)))))
#define LIBXSMM_SETENUM(DST, INP, OUT) DST = LIBXSMM_GETENUM(INP, OUT)

/* Construct an enumerator (libxsmm_datatype) from a built-in type (float, double, etc.). */
#define LIBXSMM_DATATYPE(TYPE) LIBXSMM_CONCATENATE(LIBXSMM_DATATYPE_, LIBXSMM_TYPESYMBOL(TYPE))
/* Construct a type-id from built-in input/output types (float, double, etc.). */
#define LIBXSMM_DATATYPE2(ITYPE, OTYPE) (libxsmm_datatype)LIBXSMM_GETENUM( \
  LIBXSMM_DATATYPE(ITYPE), LIBXSMM_DATATYPE(OTYPE))

/** Maximum size available to store a descriptor/blob (GEMM, MCOPY, TRANS, TRSM, TRMM). */
#if !defined(LIBXSMM_DESCRIPTOR_MAXSIZE)
# define LIBXSMM_DESCRIPTOR_MAXSIZE 96
#endif
/** Size of the descriptor considered as unique/small signature. */
#if !defined(LIBXSMM_DESCRIPTOR_SIGSIZE)
# if defined(LIBXSMM_UNPACKED)
#   define LIBXSMM_DESCRIPTOR_SIGSIZE 64
# else
#   define LIBXSMM_DESCRIPTOR_SIGSIZE 32
# endif
#endif

/** Integer type used to represent tick of a high-resolution timer. */
typedef unsigned long long libxsmm_timer_tickint;

/** Special type for bitfield flags. */
typedef unsigned int libxsmm_bitfield;

/**
 * Support for low-precision types.
 * TODO: rely on struct for proper
 * overload in C++.
 */
typedef unsigned short libxsmm_bfloat16;
typedef unsigned char  libxsmm_bfloat8;
typedef unsigned char  libxsmm_hfloat8;
typedef unsigned short libxsmm_float16;

LIBXSMM_EXTERN_C typedef union libxsmm_float_uint {
  float f;
  unsigned int u;
} libxsmm_float_uint;

LIBXSMM_EXTERN_C typedef union libxsmm_float16_ushort {
  libxsmm_float16 f;
  unsigned short  u;
} libxsmm_float16_ushort;

LIBXSMM_EXTERN_C typedef union libxsmm_bfloat16_f32 {
  libxsmm_bfloat16 i[2];
  float f;
} libxsmm_bfloat16_f32;

LIBXSMM_EXTERN_C typedef union libxsmm_bfloat8_f16 {
  libxsmm_bfloat8 i[2];
  libxsmm_float16 hf;
} libxsmm_bfloat8_f16;

#if defined(__cplusplus)
namespace Eigen { struct bfloat16; }
#endif /*__cplusplus*/

/** Integer type for LAPACK/BLAS (LP64: 32-bit, and ILP64: 64-bit). */
typedef LIBXSMM_BLASINT libxsmm_blasint;

/** Type representing sufficient storage space for a GEMM handle. */
LIBXSMM_EXTERN_C typedef struct libxsmm_gemm_blob { char data[128]; } libxsmm_gemm_blob;

/** Type representing sufficient storage space for descriptors (GEMM, TCOPY, MCOPY). */
LIBXSMM_EXTERN_C typedef struct libxsmm_descriptor_blob {
  char data[LIBXSMM_DESCRIPTOR_MAXSIZE];
} libxsmm_descriptor_blob;

/** Structure storing arguments of GEMM-like routines. */
LIBXSMM_EXTERN_C typedef struct libxsmm_gemm_descriptor libxsmm_gemm_descriptor;
/** Structure storing arguments of the matrix-eltw routine. */
LIBXSMM_EXTERN_C typedef struct libxsmm_meltw_descriptor libxsmm_meltw_descriptor;
/** Structure storing arguments of the matrix-equation routine. */
LIBXSMM_EXTERN_C typedef struct libxsmm_meqn_descriptor libxsmm_meqn_descriptor;

/**
 * Enumerates primitive element/data types.
 * Related: LIBXSMM_TYPESIZE, LIBXSMM_TYPEINFO,
 * and LIBXSMM_TYPENAME.
 */
typedef enum libxsmm_datatype {
  LIBXSMM_DATATYPE_F64,
  LIBXSMM_DATATYPE_F32,
  LIBXSMM_DATATYPE_BF16,
  LIBXSMM_DATATYPE_F16,
  LIBXSMM_DATATYPE_BF8,
  LIBXSMM_DATATYPE_HF8,
  LIBXSMM_DATATYPE_I64,
  LIBXSMM_DATATYPE_U64,
  LIBXSMM_DATATYPE_I32,
  LIBXSMM_DATATYPE_U32,
  LIBXSMM_DATATYPE_I16,
  LIBXSMM_DATATYPE_U16,
  LIBXSMM_DATATYPE_I8,
  LIBXSMM_DATATYPE_U8,
  LIBXSMM_DATATYPE_IMPLICIT,
  LIBXSMM_DATATYPE_UNSUPPORTED
} libxsmm_datatype;

typedef enum libxsmm_meltw_operation {
  LIBXSMM_MELTW_OPERATION_NONE              =  0,
  LIBXSMM_MELTW_OPERATION_UNARY             =  1,
  LIBXSMM_MELTW_OPERATION_BINARY            =  2,
  LIBXSMM_MELTW_OPERATION_TERNARY           =  3
} libxsmm_meltw_operation;

typedef enum libxsmm_meltw_unary_flags {
  LIBXSMM_MELTW_FLAG_UNARY_NONE               = 0,
  LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT  = 1,
  LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW          = 2,
  LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL          = 4,
  LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR       = 8,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS        = 16,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS        = 32,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC    = 64,
  LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES    = 128,
  LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES    = 256,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INF_ACC     = 512,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_NO_PREFETCH = 1024,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_RECORD_ARGOP= 2048,
  LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND   = 4096,
  LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS            = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
  LIBXSMM_MELTW_FLAG_UNARY_GS_COLS            = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS,
  LIBXSMM_MELTW_FLAG_UNARY_GS_OFFS            = 8192,
  LIBXSMM_MELTW_FLAG_UNARY_NTS_HINT           = 16384,
  LIBXSMM_MELTW_FLAG_UNARY_NO_SCF_QUANT       = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_NO_PREFETCH,
  LIBXSMM_MELTW_FLAG_UNARY_SIGN_SAT_QUANT     = LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS
} libxsmm_meltw_unary_flags;

typedef enum libxsmm_meltw_unary_type {
  LIBXSMM_MELTW_TYPE_UNARY_NONE                         =  0,
  LIBXSMM_MELTW_TYPE_UNARY_IDENTITY                     =  1,  /* this is copy */
  LIBXSMM_MELTW_TYPE_UNARY_XOR                          =  2,  /* this is zero */
  LIBXSMM_MELTW_TYPE_UNARY_X2                           =  3,
  LIBXSMM_MELTW_TYPE_UNARY_SQRT                         =  4,
  LIBXSMM_MELTW_TYPE_UNARY_RELU                         =  5,
  LIBXSMM_MELTW_TYPE_UNARY_RELU_INV                     =  6,
  LIBXSMM_MELTW_TYPE_UNARY_TANH                         =  7,
  LIBXSMM_MELTW_TYPE_UNARY_TANH_INV                     =  8,
  LIBXSMM_MELTW_TYPE_UNARY_SIGMOID                      =  9,
  LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV                  = 10,
  LIBXSMM_MELTW_TYPE_UNARY_GELU                         = 11,
  LIBXSMM_MELTW_TYPE_UNARY_GELU_INV                     = 12,
  LIBXSMM_MELTW_TYPE_UNARY_NEGATE                       = 13,
  LIBXSMM_MELTW_TYPE_UNARY_INC                          = 14,
  LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL                   = 15,
  LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT              = 16,
  LIBXSMM_MELTW_TYPE_UNARY_EXP                          = 17,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD              = 18,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD             = 19,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD           = 20,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX              = 21,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MUL              = 22,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD_NCNC_FORMAT  = 23,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_TO_SCALAR_OP_ADD      = 24,
  LIBXSMM_MELTW_TYPE_UNARY_DROPOUT                      = 25,
  LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV                  = 26,
  LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR            = 27,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2      = 28,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT      = 29,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T    = 30,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T     = 31,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD  = 32,
  LIBXSMM_MELTW_TYPE_UNARY_UNZIP                        = 33,
  LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU                   = 34,
  LIBXSMM_MELTW_TYPE_UNARY_LEAKY_RELU_INV               = 35,
  LIBXSMM_MELTW_TYPE_UNARY_ELU                          = 36,
  LIBXSMM_MELTW_TYPE_UNARY_ELU_INV                      = 37,
  LIBXSMM_MELTW_TYPE_UNARY_STOCHASTIC_ROUND             = 38,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2          = 39,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2          = 40,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2         = 41,
  LIBXSMM_MELTW_TYPE_UNARY_QUANT                        = 42,
  LIBXSMM_MELTW_TYPE_UNARY_DEQUANT                      = 43,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_ADD       = 44,
  LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_1   = 45,
  LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_2   = 46,
  LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_4   = 47,
  LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_8   = 48,
  LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_16  = 49,
  LIBXSMM_MELTW_TYPE_UNARY_DECOMPRESS_SPARSE_FACTOR_32  = 50,
  LIBXSMM_MELTW_TYPE_UNARY_GATHER                       = 51,
  LIBXSMM_MELTW_TYPE_UNARY_SCATTER                      = 52,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX       = 53,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4      = 54,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T    = 55,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T     = 56,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD  = 57,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4          = 58,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4          = 59,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4         = 60,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM      = 61,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2     = 62,
  LIBXSMM_MELTW_TYPE_UNARY_DUMP                         = 63,
  LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2        = 64,
  LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3        = 65,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4T_TO_NORM     = 66,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2T_TO_NORM     = 67,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MIN       = 68,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MIN              = 69,
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ABSMAX           = 70,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8      = 71,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T    = 72,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8T     = 73,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8_PAD  = 74,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8T_TO_NORM     = 75,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_NORM      = 76
} libxsmm_meltw_unary_type;

typedef enum libxsmm_meltw_binary_flags {
  LIBXSMM_MELTW_FLAG_BINARY_NONE              = 0,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0    = 1,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1    = 2,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0    = 4,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1    = 8,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0 = 16,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1 = 32,
  LIBXSMM_MELTW_FLAG_BINARY_STOCHASTIC_ROUND  = 64,
  LIBXSMM_MELTW_FLAG_BINARY_BITMASK_2BYTEMULT = 128,
  LIBXSMM_MELTW_FLAG_BINARY_NTS_HINT          = 256
} libxsmm_meltw_binary_flags;

typedef enum libxsmm_meltw_binary_type {
  LIBXSMM_MELTW_TYPE_BINARY_NONE                            =  0,
  LIBXSMM_MELTW_TYPE_BINARY_ADD                             =  1,
  LIBXSMM_MELTW_TYPE_BINARY_MUL                             =  2,
  LIBXSMM_MELTW_TYPE_BINARY_SUB                             =  3,
  LIBXSMM_MELTW_TYPE_BINARY_DIV                             =  4,
  LIBXSMM_MELTW_TYPE_BINARY_MULADD                          =  5,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL                          =  6,
  LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD =  7,
  LIBXSMM_MELTW_TYPE_BINARY_PACK                            =  8,
  LIBXSMM_MELTW_TYPE_BINARY_MAX                             =  9,
  LIBXSMM_MELTW_TYPE_BINARY_MIN                             = 10,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM                          = 11,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_B_TRANS                  = 12,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_TRANS                  = 13,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_TRANS_B_TRANS          = 14,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI                   = 15,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_B_TRANS           = 16,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_TRANS             = 17,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_TRANS_B_TRANS     = 18,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_B_TRANS                  = 19,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_TRANS                  = 20,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_TRANS_B_TRANS          = 21,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI                   = 22,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI_B_TRANS           = 23,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI_TRANS             = 24,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI_TRANS_B_TRANS     = 25,
  LIBXSMM_MELTW_TYPE_BINARY_ZIP                             = 26,
  LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GT                       = 27,
  LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GE                       = 28,
  LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT                       = 29,
  LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE                       = 30,
  LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_EQ                       = 31,
  LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE                       = 32
} libxsmm_meltw_binary_type;

typedef enum libxsmm_meltw_ternary_flags {
  LIBXSMM_MELTW_FLAG_TERNARY_NONE              =  0,
  LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0    =  1,
  LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1    =  2,
  LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2    =  4,
  LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0    =  8,
  LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1    =  16,
  LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2    =  32,
  LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0 =  64,
  LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 =  128,
  LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 =  256,
  LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT =  512,
  LIBXSMM_MELTW_FLAG_TERNARY_BITMASK_2BYTEMULT =  1024,
  LIBXSMM_MELTW_FLAG_TERNARY_STOCHASTIC_ROUND  =  2048
} libxsmm_meltw_ternary_flags;

typedef enum libxsmm_meltw_ternary_type {
  LIBXSMM_MELTW_TYPE_TERNARY_NONE        =  0,
  LIBXSMM_MELTW_TYPE_TERNARY_MULADD      =  1,
  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL      =  2,
  LIBXSMM_MELTW_TYPE_TERNARY_SELECT      =  3,
  LIBXSMM_MELTW_TYPE_TERNARY_NMULADD     =  4,
  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM                       =  5,
  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_B_TRANS               =  6,
  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_TRANS               =  7,
  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_TRANS_B_TRANS       =  8,
  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI                =  9,
  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI_B_TRANS        =  10,
  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI_TRANS          =  11,
  LIBXSMM_MELTW_TYPE_TERNARY_BRGEMM_A_VNNI_TRANS_B_TRANS  =  12,
  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_B_TRANS               =  13,
  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_TRANS               =  14,
  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_TRANS_B_TRANS       =  15,
  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_VNNI                =  16,
  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_VNNI_B_TRANS        =  17,
  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_VNNI_TRANS          =  18,
  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL_A_VNNI_TRANS_B_TRANS  =  19
} libxsmm_meltw_ternary_type;

/** Flag enumeration which can be binary ORed, for very simple dispatch interface */
typedef enum libxsmm_basic_gemm_flags {
  LIBXSMM_BASIC_GEMM_FLAG_NONE = 0,
  /** Transpose matrix A. */
  LIBXSMM_BASIC_GEMM_FLAG_TRANS_A = 1,
  /** Transpose matrix B. */
  LIBXSMM_BASIC_GEMM_FLAG_TRANS_B = 2,
  /** Transpose matrix A and B. */
  LIBXSMM_BASIC_GEMM_FLAG_TRANS_AB = LIBXSMM_BASIC_GEMM_FLAG_TRANS_A | LIBXSMM_BASIC_GEMM_FLAG_TRANS_B,
  /** Beta=0|1 */
  LIBXSMM_BASIC_GEMM_FLAG_BETA_0 = 4,
  /** Generate aligned load instructions. */
  LIBXSMM_BASIC_GEMM_FLAG_ALIGN_A = 8,
  /** Aligned load/store instructions. */
  LIBXSMM_BASIC_GEMM_FLAG_ALIGN_C = 16,
  /** Aligned C matrix, but using NTS Hint when storing */
  LIBXSMM_BASIC_GEMM_FLAG_ALIGN_C_NTS_HINT = 1024 | LIBXSMM_BASIC_GEMM_FLAG_ALIGN_C,
  /** Marker flag; do not use. */
  LIBXSMM_BASIC_GEMM_FLAG_INVALID = 524288
} libxsmm_basic_gemm_flags;

/** Flag enumeration which can be binary ORed. */
typedef enum libxsmm_gemm_flags {
  LIBXSMM_GEMM_FLAG_NONE = 0,
  /** Transpose matrix A. */
  LIBXSMM_GEMM_FLAG_TRANS_A = 1,
  /** Transpose matrix B. */
  LIBXSMM_GEMM_FLAG_TRANS_B = 2,
  /** Transpose matrix A and B. */
  LIBXSMM_GEMM_FLAG_TRANS_AB = LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B,
  /** Beta=0|1 */
  LIBXSMM_GEMM_FLAG_BETA_0 = 4,
  /** Generate aligned load instructions. */
  LIBXSMM_GEMM_FLAG_ALIGN_A = 8,
  /** Aligned load/store instructions. */
  LIBXSMM_GEMM_FLAG_ALIGN_C = 16,
  /** Batch-reduce Ai * Bi. */
  /** Aligned C matrix, but using NTS Hint when storing */
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT = 32 | LIBXSMM_GEMM_FLAG_ALIGN_C,
  /** AMX hint to avoid tileconfig/release, it's negated bits, so that 0 is default "on" */
  LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG = 64,
  LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG = 128,
  /* in case of integer GEMM, if A is unsigned */
  LIBXSMM_GEMM_FLAG_A_UNSIGNED = 256,
  /* in case of integer GEMM, if B is unsigned */
  LIBXSMM_GEMM_FLAG_B_UNSIGNED = 512,
  /* in case of integer GEMM, if C is unsigned */
  LIBXSMM_GEMM_FLAG_C_UNSIGNED = 1024,
  /* in case of integer GEMM, if A and B are unsigned */
  LIBXSMM_GEMM_FLAG_AB_UNSIGNED = LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_B_UNSIGNED,
  /* for low precision we also require up-front packed formats "VNNI" for best performance, this flag indicates A */
  LIBXSMM_GEMM_FLAG_VNNI_A = 2048,
  /* for low precision we also require up-front packed formats "VNNI" for best performance, this flag indicates B */
  LIBXSMM_GEMM_FLAG_VNNI_B = 4096,
  /* for low precision we also require post packed formats "VNNI" for best performance, this flag indicated C */
  LIBXSMM_GEMM_FLAG_VNNI_C = 8192,
  /** use GEMM ABI */
  LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI = 16384,
  /** use XGEMM_EXT ABI */
  LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI = 32768,

  /* Pseudo-flag denoting big descriptor */
  LIBXSMM_GEMM_FLAG_DESC_ISBIG = 65536,
  /** Batch-reduce Ai * Bi. */
  LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS = 65536,
  /** Batch-reduce Ai * Bi. */
  LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET = 131072,
  /** Batch-reduce Ai * Bi. */
  LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE = 262144,
  LIBXSMM_GEMM_FLAG_USE_COL_VEC_SCF = 524288,
  LIBXSMM_GEMM_FLAG_USE_COL_VEC_ZPT = 1048576,
  LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI2 = 2097152,
  LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_INT4_VNNI8_INTLV = 4194304,
  LIBXSMM_GEMM_FLAG_DECOMPRESS_A_VIA_BITMASK = 8388608,
  LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI2 = 16777216,
  LIBXSMM_GEMM_FLAG_USE_MxK_ZPT = 33554432,
  LIBXSMM_GEMM_FLAG_USE_MxK_SCF = 67108864,
  LIBXSMM_GEMM_FLAG_INTERPRETE_A_AS_MXFP4_VNNI8_INTLV = 134217728,
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
  LIBXSMM_GEMM_FLAG_INVALID = 268435456
} libxsmm_gemm_flags;

/** Enumeration of the available prefetch strategies. */
typedef enum libxsmm_gemm_prefetch_type {
  /** No data-prefetch. */
  LIBXSMM_GEMM_PREFETCH_NONE               = LIBXSMM_PREFETCH_NONE,
  /** Prefetch PA using accesses to A. */
  LIBXSMM_GEMM_PREFETCH_AL2                = 1,
  /** Prefetch PA (aggressive). */
  LIBXSMM_GEMM_PREFETCH_BL2_VIA_C          = 2,
  /** Prefetch A ahead. */
  LIBXSMM_GEMM_PREFETCH_AL2_AHEAD          = 4,
  LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C       = LIBXSMM_GEMM_PREFETCH_BL2_VIA_C | LIBXSMM_GEMM_PREFETCH_AL2,
  LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD = LIBXSMM_GEMM_PREFETCH_BL2_VIA_C | LIBXSMM_GEMM_PREFETCH_AL2_AHEAD,
  /** Backward compatibility: AL2CL2BL2_VIA_C is an alias for AL2BL2_VIA_C (Eigen library). */
  LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C         = LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C,
  /** Current B into L1. */
  LIBXSMM_GEMM_PREFETCH_BL1                = 8,
  LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB         = 16,
  LIBXSMM_GEMM_PREFETCH_C_SCRATCH          = 32,
  LIBXSMM_GEMM_PREFETCH_C                  = 64
} libxsmm_gemm_prefetch_type;

/** Enumeration of the batchreduce type. */
typedef enum libxsmm_gemm_batch_reduce_type {
  LIBXSMM_GEMM_BATCH_REDUCE_NONE    = 0,
  LIBXSMM_GEMM_BATCH_REDUCE_ADDRESS = 1,
  LIBXSMM_GEMM_BATCH_REDUCE_OFFSET  = 2,
  LIBXSMM_GEMM_BATCH_REDUCE_STRIDE  = 4
} libxsmm_gemm_batch_reduce_type;

/** Determines the kernel kind. */
typedef enum libxsmm_kernel_kind {
  /** Matrix multiplication kernel */
  LIBXSMM_KERNEL_KIND_MATMUL  = 0,
  /** Mateltw kernel kind */
  LIBXSMM_KERNEL_KIND_MELTW   = 1,
  /** Mateqn kernel kind */
  LIBXSMM_KERNEL_KIND_MEQN    = 2,
  /** User-defined kernels */
  LIBXSMM_KERNEL_KIND_USER    = 3,
  /** Not a JIT kernel */
  LIBXSMM_KERNEL_UNREGISTERED = 4
} libxsmm_kernel_kind;

LIBXSMM_EXTERN_C typedef struct libxsmm_matrix_arg {
  void* primary;
  void* secondary;
  void* tertiary;
  void* quaternary;
} libxsmm_matrix_arg;

LIBXSMM_EXTERN_C typedef struct libxsmm_matrix_op_arg {
  void* primary;
  void* secondary;
  void* tertiary;
  void* quaternary;
} libxsmm_matrix_op_arg;

LIBXSMM_EXTERN_C typedef struct libxsmm_meqn_arg_shape {
  libxsmm_blasint m;                    /* number of rows */
  libxsmm_blasint n;                    /* number of cols */
  libxsmm_blasint ld;                   /* leading dimension of input */
  libxsmm_datatype type;                /* datatype of input */
} libxsmm_meqn_arg_shape;

LIBXSMM_EXTERN_C typedef struct libxsmm_meltw_unary_shape {
  libxsmm_blasint m;                    /* number of rows */
  libxsmm_blasint n;                    /* number of cols */
  libxsmm_blasint ldi;                  /* leading dimension of first input */
  libxsmm_blasint ldo;                  /* leading dimension of output */
  libxsmm_datatype in0_type;            /* datatype of input */
  libxsmm_datatype out_type;            /* datatype of output */
  libxsmm_datatype comp_type;           /* datatype of compute */
} libxsmm_meltw_unary_shape;

LIBXSMM_EXTERN_C typedef struct libxsmm_meltw_binary_shape {
  libxsmm_blasint m;                    /* number of rows */
  libxsmm_blasint n;                    /* number of cols */
  libxsmm_blasint ldi;                  /* leading dimension of first input */
  libxsmm_blasint ldi2;                 /* leading dimension of second input */
  libxsmm_blasint ldo;                  /* leading dimension of output */
  libxsmm_datatype in0_type;            /* datatype of input 0 */
  libxsmm_datatype in1_type;            /* datatype of input 1 */
  libxsmm_datatype out_type;            /* datatype of output */
  libxsmm_datatype comp_type;           /* datatype of compute */
} libxsmm_meltw_binary_shape;

LIBXSMM_EXTERN_C typedef struct libxsmm_meltw_ternary_shape {
  libxsmm_blasint m;                    /* number of rows */
  libxsmm_blasint n;                    /* number of cols */
  libxsmm_blasint ldi;                  /* leading dimension of first input */
  libxsmm_blasint ldi2;                 /* leading dimension of second input */
  libxsmm_blasint ldi3;                 /* leading dimension of third input */
  libxsmm_blasint ldo;                  /* leading dimension of output */
  libxsmm_datatype in0_type;            /* datatype of input 0 */
  libxsmm_datatype in1_type;            /* datatype of input 1 */
  libxsmm_datatype in2_type;            /* datatype of input 2 */
  libxsmm_datatype out_type;            /* datatype of output */
  libxsmm_datatype comp_type;           /* datatype of compute */
} libxsmm_meltw_ternary_shape;

typedef enum libxsmm_matrix_arg_type {
  LIBXSMM_MATRIX_ARG_TYPE_SINGULAR = 0,
  LIBXSMM_MATRIX_ARG_TYPE_SET      = 1
} libxsmm_matrix_arg_type;

typedef enum libxsmm_matrix_arg_set_type {
  LIBXSMM_MATRIX_ARG_SET_TYPE_NONE        = 0,
  LIBXSMM_MATRIX_ARG_SET_TYPE_ABS_ADDRESS = 1,
  LIBXSMM_MATRIX_ARG_SET_TYPE_OFFSET_BASE = 2,
  LIBXSMM_MATRIX_ARG_SET_TYPE_STRIDE_BASE = 3
} libxsmm_matrix_arg_set_type;

LIBXSMM_EXTERN_C typedef struct libxsmm_matrix_arg_attributes {
  libxsmm_matrix_arg_type     type;
  libxsmm_matrix_arg_set_type set_type;
  libxsmm_blasint             set_cardinality_hint;
  libxsmm_blasint             set_stride_hint;
} libxsmm_matrix_arg_attributes;

LIBXSMM_EXTERN_C typedef struct libxsmm_meqn_op_metadata {
  libxsmm_blasint eqn_idx;
  libxsmm_blasint op_arg_pos;
} libxsmm_meqn_op_metadata;

LIBXSMM_EXTERN_C typedef struct libxsmm_meqn_arg_metadata {
  libxsmm_blasint eqn_idx;
  libxsmm_blasint in_arg_pos;
} libxsmm_meqn_arg_metadata;

/** argument struct for matrix-eltwise: unary */
LIBXSMM_EXTERN_C typedef struct libxsmm_meltw_unary_param {
  libxsmm_matrix_op_arg op;   /* op state & parameters */
  libxsmm_matrix_arg in;      /* input  */
  libxsmm_matrix_arg out;     /* output */
} libxsmm_meltw_unary_param;

/** argument struct for matrix-eltwise: binary */
LIBXSMM_EXTERN_C typedef struct libxsmm_meltw_binary_param {
  libxsmm_matrix_op_arg op;   /* op state & parameters */
  libxsmm_matrix_arg in0;     /* 1st input  */
  libxsmm_matrix_arg in1;     /* 2nd input  */
  libxsmm_matrix_arg out;     /* output     */
} libxsmm_meltw_binary_param;

/** argument struct for matrix-eltwise: ternary */
LIBXSMM_EXTERN_C typedef struct libxsmm_meltw_ternary_param {
  libxsmm_matrix_op_arg op;   /* op state & parameters */
  libxsmm_matrix_arg in0;     /* 1st input  */
  libxsmm_matrix_arg in1;     /* 2nd input  */
  libxsmm_matrix_arg in2;     /* 3rd input  */
  libxsmm_matrix_arg out;     /* output     */
} libxsmm_meltw_ternary_param;

/** argument struct for matrix equation */
LIBXSMM_EXTERN_C typedef struct libxsmm_meqn_param {
  const libxsmm_matrix_op_arg* ops_args;    /* op state & parameters */
  const libxsmm_matrix_arg*    inputs;      /* array of input args */
  libxsmm_matrix_arg           output;      /* output arg */
} libxsmm_meqn_param;

/** Specialized function for matrix-eltw (weak-typed). */
LIBXSMM_EXTERN_C typedef void (*libxsmm_meltwfunction_unary)(const libxsmm_meltw_unary_param* in_struct);
LIBXSMM_EXTERN_C typedef void (*libxsmm_meltwfunction_binary)(const libxsmm_meltw_binary_param* in_struct);
LIBXSMM_EXTERN_C typedef void (*libxsmm_meltwfunction_ternary)(const libxsmm_meltw_ternary_param* in_struct);
/* matrix equation function */
LIBXSMM_EXTERN_C typedef void (*libxsmm_meqn_function)(const libxsmm_meqn_param* in_struct);

LIBXSMM_EXTERN_C typedef union libxsmm_xmeltwfunction {
  void (*xmeltw)(const void* in_struct);
  libxsmm_meltwfunction_unary meltw_unary;
  libxsmm_meltwfunction_binary meltw_binary;
  libxsmm_meltwfunction_ternary meltw_ternary;
} libxsmm_xmeltwfunction;

/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (double-precision). */
LIBXSMM_EXTERN_C typedef void (*libxsmm_dmmfunction)(const double* a, const double* b, double* c);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (single-precision). */
LIBXSMM_EXTERN_C typedef void (*libxsmm_smmfunction)(const float* a, const float* b, float* c);

/* argument structs for generalized interface */
LIBXSMM_EXTERN_C typedef struct libxsmm_gemm_param {
  libxsmm_matrix_op_arg op;  /* op state & parameters */
  libxsmm_matrix_arg a;   /* a matrix  */
  libxsmm_matrix_arg b;   /* b matrix  */
  libxsmm_matrix_arg c;   /* c matrix  */
} libxsmm_gemm_param;

LIBXSMM_EXTERN_C typedef struct libxsmm_gemm_ext_param {
  libxsmm_matrix_op_arg op;  /* op state & parameters */
  libxsmm_matrix_arg a;   /* a matrix  */
  libxsmm_matrix_arg b;   /* b matrix  */
  libxsmm_matrix_arg c;   /* c matrix  */
  libxsmm_matrix_arg d;   /* additional tensor for binary op on c */
  libxsmm_matrix_arg ap;  /* a after applying unary op */
  libxsmm_matrix_arg bp;  /* b after applying unary op */
  libxsmm_matrix_arg cp;  /* c before applying binary/ternary op after GEMM */
} libxsmm_gemm_ext_param;

LIBXSMM_EXTERN_C typedef struct libxsmm_gemm_shape {
  libxsmm_blasint m;                    /* number of rows of A and C */
  libxsmm_blasint n;                    /* number of cols of C and B */
  libxsmm_blasint k;                    /* number of cols of A and number of rows of B */
  libxsmm_blasint lda;                  /* leading dimension of A */
  libxsmm_blasint ldb;                  /* leading dimension of B */
  libxsmm_blasint ldc;                  /* leading dimension of C */
  libxsmm_datatype a_in_type;           /* datatype of A */
  libxsmm_datatype b_in_type;           /* datatype of B */
  libxsmm_datatype out_type;            /* datatype of C */
  libxsmm_datatype comp_type;           /* datatype of inner product */
} libxsmm_gemm_shape;

LIBXSMM_EXTERN_C typedef struct libxsmm_gemm_batch_reduce_config {
  libxsmm_gemm_batch_reduce_type br_type;  /* specifying the type of the BRGEMM operation */
  libxsmm_blasint br_stride_a_hint;        /* mandatory hint for strided BRGEMM */
  libxsmm_blasint br_stride_b_hint;        /* mandatory hint for strided BRGEMM */
  unsigned char br_unroll_hint;            /* optional hint containing the BR count */
} libxsmm_gemm_batch_reduce_config;

LIBXSMM_EXTERN_C typedef struct libxsmm_spgemm_config {
  libxsmm_blasint packed_width;        /* Packed width for packed spgemm */
  libxsmm_blasint bk;                  /* Bk size for dense block        */
  libxsmm_blasint bn;                  /* Bn size for dense block        */
} libxsmm_spgemm_config;

LIBXSMM_EXTERN_C typedef struct libxsmm_gemm_ext_unary_argops {
  libxsmm_blasint ldap;                       /* leading dimensions of Ap */
  libxsmm_meltw_unary_type ap_unary_type;     /* op type for Ap = unary( A ) */
  libxsmm_bitfield ap_unary_flags;            /* flags for Ap = unary( A ) */
  libxsmm_blasint store_ap;                   /* nonzero for storing Ap */
  libxsmm_blasint ldbp;                       /* leading dimensions of Bp */
  libxsmm_meltw_unary_type bp_unary_type;     /* op type for Bp = unary( B ) */
  libxsmm_bitfield bp_unary_flags;            /* flags for Bp = unary( B ) */
  libxsmm_blasint store_bp;                   /* nonzero for storing Bp */
  libxsmm_blasint ldcp;                       /* leading dimensions of Cp */
  libxsmm_meltw_unary_type cp_unary_type;     /* op type for Cp = unary( C ) */
  libxsmm_bitfield cp_unary_flags;           /* flags for Cp = unary( C ) */
  libxsmm_blasint store_cp;                   /* nonzero for storing Cp */
} libxsmm_gemm_ext_unary_argops;

LIBXSMM_EXTERN_C typedef struct libxsmm_gemm_ext_binary_postops {
  libxsmm_blasint ldd;                        /* leading dimensions of D */
  libxsmm_datatype d_in_type;                 /* datatype of D */
  libxsmm_meltw_binary_type d_binary_type;    /* op type for C = binary( C, D ) */
  libxsmm_bitfield d_binary_flags;            /* flags for C = binary( C, D ) */
} libxsmm_gemm_ext_binary_postops;

LIBXSMM_EXTERN_C typedef struct libxsmm_tilecfg_state {
  unsigned char tileconfig[64];
} libxsmm_tilecfg_state;

/* generalized and extended functions for everything that is not a basic GEMM as defined above */
LIBXSMM_EXTERN_C typedef void (*libxsmm_gemmfunction)    ( const libxsmm_gemm_param*     in_struct );
LIBXSMM_EXTERN_C typedef void (*libxsmm_gemmfunction_ext)( const libxsmm_gemm_ext_param* in_struct );
LIBXSMM_EXTERN_C typedef void (*libxsmm_tilecfgfunction) ( const libxsmm_tilecfg_state*  in_struct );

/** Union to convert between different function types or plain pointers (weak-typed). */
LIBXSMM_EXTERN_C typedef union libxsmm_xmmfunction {
  const void* ptr_const; void* ptr;
  void (*xmm)(const void* a, const void* b, void* c);
  void (*xgemm)(const void* in_struct);
  libxsmm_dmmfunction dmm; libxsmm_smmfunction smm;
  libxsmm_gemmfunction gemm; libxsmm_gemmfunction_ext gemm_ext;
  libxsmm_tilecfgfunction tilecfg;
} libxsmm_xmmfunction;

/** Structure to receive information about GEMM-kernels (libxsmm_get_mmkernel_info). */
LIBXSMM_EXTERN_C typedef struct libxsmm_mmkernel_info {
  /** Input/output data-type */
  libxsmm_datatype iprecision, oprecision;
  /** Prefetch strategy. */
  libxsmm_gemm_prefetch_type prefetch;
  /** Leading dimensions. */
  unsigned int lda, ldb, ldc;
  /** Extents/shape. */
  unsigned int m, n, k;
  /** Set of flags. */
  int flags;
} libxsmm_mmkernel_info;

/** Structure to receive information about matrix-eltw kernels (libxsmm_get_meltwkernel_info). */
LIBXSMM_EXTERN_C typedef struct libxsmm_meltwkernel_info {
  /** LDx, M, and N. */
  unsigned int ldi, ldo, m, n;
  /** Size of data element. */
  unsigned int datatype;
  /** Set of flags. */
  unsigned int flags;
  /** Set of operation. */
  unsigned int operation;
} libxsmm_meltwkernel_info;

LIBXSMM_EXTERN_C typedef struct libxsmm_kernel_info {
  libxsmm_kernel_kind kind;
  /** Number of FLoating Point OperationS (FLOPS). */
  unsigned int nflops;
  /** Code size (Bytes). */
  size_t code_size;
} libxsmm_kernel_info;

/** Structure to receive information about the code registry status (libxsmm_get_registry_info). */
LIBXSMM_EXTERN_C typedef struct libxsmm_registry_info {
  size_t capacity, size, nbytes, nstatic, ncache;
} libxsmm_registry_info;

#endif /*LIBXSMM_TYPEDEFS_H*/
