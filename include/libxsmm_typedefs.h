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

/** Generic prefetches; similar to LIBXSMM_PREFETCH_AUTO (libxsmm_frontend.h) */
#define LIBXSMM_PREFETCH_SIGONLY 1
#define LIBXSMM_PREFETCH_NONE 0

/** Helper macro for type names. */
#define LIBXSMM_TYPENAME(TYPE) LIBXSMM_STRINGIFY(LIBXSMM_CONCATENATE(LIBXSMM_TYPENAME_, TYPE))
#define LIBXSMM_TYPENAME_double f64
#define LIBXSMM_TYPENAME_float f32
#define LIBXSMM_TYPENAME_libxsmm_bfloat16 bf16
#define LIBXSMM_TYPENAME_libxsmm_float16 f16
#define LIBXSMM_TYTPNAME_libxsmm_bfloat8 bf8
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
#define LIBXSMM_TYPESYMBOL_int I32
#define LIBXSMM_TYPESYMBOL_short I16
#define LIBXSMM_TYPESYMBOL_char I8

#define LIBXSMM_TYPESIZE(ENUM) ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_F64  ? 8 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_F32  ? 4 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_BF16 ? 2 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_F16  ? 2 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_BF8  ? 1 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_I64  ? 8 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_I32  ? 4 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_I16  ? 2 : ( \
  ((int)(ENUM)) == LIBXSMM_DATATYPE_I8   ? 1 : ( \
  0/*invalid*/))))))))))

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

/* special type for bitfield flags */
typedef unsigned int libxsmm_bitfield;

/* Support for Bfloat16 */
typedef unsigned short libxsmm_bfloat16;
typedef unsigned char  libxsmm_bfloat8;
typedef unsigned short libxsmm_float16;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_float_uint {
  float f;
  unsigned int u;
} libxsmm_float_uint;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_bfloat16_hp {
  libxsmm_bfloat16 i[2];
  float f;
} libxsmm_bfloat16_hp;

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_bfloat8_qp {
  libxsmm_bfloat8 i[2];
  libxsmm_float16 hf;
} libxsmm_bfloat8_qp;

#if defined(__cplusplus)
namespace Eigen { struct bfloat16; }
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
/** Structure storing arguments of the matrix-eltw routine. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meltw_descriptor libxsmm_meltw_descriptor;
/** Structure storing arguments of the matrix-equation routine. */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meqn_descriptor libxsmm_meqn_descriptor;

/** Enumerates element/data types. */
typedef enum libxsmm_datatype {
  LIBXSMM_DATATYPE_F64,
  LIBXSMM_DATATYPE_F32,
  LIBXSMM_DATATYPE_BF16,
  LIBXSMM_DATATYPE_F16,
  LIBXSMM_DATATYPE_BF8,
  LIBXSMM_DATATYPE_I64,
  LIBXSMM_DATATYPE_I32,
  LIBXSMM_DATATYPE_I16,
  LIBXSMM_DATATYPE_I8,
  LIBXSMM_DATATYPE_UNSUPPORTED
} libxsmm_datatype;

typedef enum libxsmm_meltw_operation {
  LIBXSMM_MELTW_OPERATION_NONE                                             =  0,
  LIBXSMM_MELTW_OPERATION_OPREDUCE_VECS_IDX                                =  1,
  LIBXSMM_MELTW_OPERATION_UNARY                                            =  2,
  LIBXSMM_MELTW_OPERATION_BINARY                                           =  3,
  LIBXSMM_MELTW_OPERATION_TERNARY                                          =  4
} libxsmm_meltw_operation;

typedef enum libxsmm_meltw_opreduce_vecs_flags {
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_NONE                           = 0,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIN_VECIDX           = 1,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN           = 2,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY                        = 4,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_ADD                         = 8,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_SUB                         = 16,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_MUL                         = 32,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DIV                         = 64,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DOT                         = 128,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_SCALE_OP_RESULT                = 256,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_NONE                     = 512,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM                      = 1024,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX                      = 2048,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MIN                      = 4096,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_INDEXED_VEC                    = 8192,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VEC           = 16384,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VECIDX        = 32768,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_0         = 65536,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_1         = 131072,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY_REDOP_SUM              = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_MUL_REDOP_SUM               = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_MUL  | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY_REDOP_MAX              = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY_REDOP_MIN              = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MIN,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDUCE_MAX_IDX_COLS            = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX,
  LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDUCE_MAX_IDX_COLS_ARGOP      = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_RECORD_ARGOP_OFF_VEC_0 | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN | LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX
} libxsmm_meltw_opreduce_vecs_flags;

typedef enum libxsmm_meltw_unary_flags {
  LIBXSMM_MELTW_FLAG_UNARY_NONE               = 0,
  LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT  = 1,
  LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW          = 2,
  LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL          = 4,
  LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR       = 8,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS        = 16,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS        = 32,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_XOR_ACC     = 64,
  LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_4BYTES    = 128,
  LIBXSMM_MELTW_FLAG_UNARY_IDX_SIZE_8BYTES    = 256,
  LIBXSMM_MELTW_FLAG_UNARY_GS_ROWS            = 512,
  LIBXSMM_MELTW_FLAG_UNARY_GS_COLS            = 1024,
  LIBXSMM_MELTW_FLAG_UNARY_GS_OFFS            = 2048,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_NEG_INF_ACC = 4096,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_RECORD_ARGOP= 8192,
  LIBXSMM_MELTW_FLAG_UNARY_REDUCE_NO_PREFETCH = 16384
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
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI       = 28,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT      = 29,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT      = 30,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNIT      = 31,
  LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI_PAD   = 32,
  LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS             = 33,
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
  LIBXSMM_MELTW_TYPE_UNARY_REDUCE_COLS_IDX_OP_MAX       = 53
} libxsmm_meltw_unary_type;

typedef enum libxsmm_meltw_binary_flags {
  LIBXSMM_MELTW_FLAG_BINARY_NONE              = 0,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0    = 1,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1    = 2,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0    = 4,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1    = 8,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0 = 16,
  LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1 = 32
} libxsmm_meltw_binary_flags;

typedef enum libxsmm_meltw_binary_type {
  LIBXSMM_MELTW_TYPE_BINARY_NONE        =  0,
  LIBXSMM_MELTW_TYPE_BINARY_ADD         =  1,
  LIBXSMM_MELTW_TYPE_BINARY_MUL         =  2,
  LIBXSMM_MELTW_TYPE_BINARY_SUB         =  3,
  LIBXSMM_MELTW_TYPE_BINARY_DIV         =  4,
  LIBXSMM_MELTW_TYPE_BINARY_MULADD      =  5,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL      =  6,
  LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD = 7,
  LIBXSMM_MELTW_TYPE_BINARY_PACK        =  8,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM                       =  9,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_B_TRANS               =  10,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_TRANS               =  11,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_TRANS_B_TRANS       =  12,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI                =  13,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_B_TRANS        =  14,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_TRANS          =  15,
  LIBXSMM_MELTW_TYPE_BINARY_BRGEMM_A_VNNI_TRANS_B_TRANS  =  16,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_B_TRANS               =  17,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_TRANS               =  18,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_TRANS_B_TRANS       =  19,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI                =  20,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI_B_TRANS        =  21,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI_TRANS          =  22,
  LIBXSMM_MELTW_TYPE_BINARY_MATMUL_A_VNNI_TRANS_B_TRANS  =  23
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
  LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT =  512
} libxsmm_meltw_ternary_flags;

typedef enum libxsmm_meltw_ternary_type {
  LIBXSMM_MELTW_TYPE_TERNARY_NONE        =  0,
  LIBXSMM_MELTW_TYPE_TERNARY_MULADD      =  1,
  LIBXSMM_MELTW_TYPE_TERNARY_MATMUL      =  2,
  LIBXSMM_MELTW_TYPE_TERNARY_BLEND       =  3,
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
  /** AMX hint to avoid tileconfig/release, it's negated bits, so that 0 is default "on" */
  LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG = 32,
  LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG = 64,
  LIBXSMM_GEMM_FLAG_BATCH_REDUCE_ADDRESS = 128,
  /** Batch-reduce Ai * Bi. */
  LIBXSMM_GEMM_FLAG_BATCH_REDUCE_OFFSET = 256,
  /** Batch-reduce Ai * Bi. */
  LIBXSMM_GEMM_FLAG_BATCH_REDUCE_STRIDE = 512,
  /** Aligned C matrix, but using NTS Hint when storing */
  LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT = 1024 | LIBXSMM_GEMM_FLAG_ALIGN_C,
  /* in case of integer GEMM, if A is unsigned */
  LIBXSMM_GEMM_FLAG_A_UNSIGNED = 2048,
  /* in case of integer GEMM, if B is unsigned */
  LIBXSMM_GEMM_FLAG_B_UNSIGNED = 4096,
  /* in case of integer GEMM, if C is unsigned */
  LIBXSMM_GEMM_FLAG_C_UNSIGNED = 8192,
  /* in case of integer GEMM, if A and B are unsigned */
  LIBXSMM_GEMM_FLAG_AB_UNSIGNED = LIBXSMM_GEMM_FLAG_A_UNSIGNED | LIBXSMM_GEMM_FLAG_B_UNSIGNED,
  /* for low precision we also require up-front packed formats "VNNI" for best performance, this flag indicates A */
  LIBXSMM_GEMM_FLAG_VNNI_A = 16384,
  /* for low precision we also require up-front packed formats "VNNI" for best performance, this flag indicates B */
  LIBXSMM_GEMM_FLAG_VNNI_B = 32768,
  /* for low precision we also require post packed formats "VNNI" for best performance, this flag indicated C */
  LIBXSMM_GEMM_FLAG_VNNI_C = 65536,
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
  /** use GEMMM ABI */
  LIBXSMM_GEMM_FLAG_USE_XGEMM_ABI = 131072,
  /** use XGEMM_EXT ABI */
  LIBXSMM_GEMM_FLAG_USE_XGEMM_EXT_ABI = 262144,
  /** Marker flag; do not use. */
  LIBXSMM_GEMM_FLAG_INVALID = 524288
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
  /** Current B into L1. */
  LIBXSMM_GEMM_PREFETCH_BL1                = 16,
  LIBXSMM_GEMM_PREFETCH_BRGEMM_OOB         = 32,
  LIBXSMM_GEMM_PREFETCH_C_SCRATCH          = 64,
  LIBXSMM_GEMM_PREFETCH_C                  = 128
} libxsmm_gemm_prefetch_type;

/** Enumeration of the batchreduce type. */
typedef enum libxsmm_gemm_batch_reduce_type {
  LIBXSMM_GEMM_BATCH_REDUCE_NONE    = 0,
  LIBXSMM_GEMM_BATCH_REDUCE_ADDRESS = 1,
  LIBXSMM_GEMM_BATCH_REDUCE_OFFSET  = 2,
  LIBXSMM_GEMM_BATCH_REDUCE_STRIDE  = 4
} libxsmm_gemm_batch_reduce_type;

/** Flag enumeration which can be binary ORed. */
typedef enum libxsmm_matcopy_flags {
  LIBXSMM_MATCOPY_FLAG_DEFAULT = 0,
  /** If set, then use zero matrix as source */
  LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE = 1
} libxsmm_matcopy_flags;

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

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_matrix_arg {
  void* primary;
  void* secondary;
  void* tertiary;
  void* quaternary;
} libxsmm_matrix_arg;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_matrix_arg_v2 {
  void* primary;
  void* secondary;
  void* tertiary;
  void* quaternary;
} libxsmm_matrix_arg_v2;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_matrix_op_arg {
  void* primary;
  void* secondary;
  void* tertiary;
  void* quaternary;
} libxsmm_matrix_op_arg;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meqn_arg_shape {
  libxsmm_blasint m;                    /* number of rows */
  libxsmm_blasint n;                    /* number of cols */
  libxsmm_blasint ld;                   /* leading dimension of input */
  libxsmm_datatype type;                /* datatype of input */
} libxsmm_meqn_arg_shape;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meltw_unary_shape {
  libxsmm_blasint m;                    /* number of rows */
  libxsmm_blasint n;                    /* number of cols */
  libxsmm_blasint ldi;                  /* leading dimension of first input */
  libxsmm_blasint ldo;                  /* leading dimension of output */
  libxsmm_datatype in0_type;            /* datatype of input */
  libxsmm_datatype out_type;            /* datatype of output */
  libxsmm_datatype comp_type;           /* datatype of compute */
} libxsmm_meltw_unary_shape;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meltw_binary_shape {
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

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meltw_ternary_shape {
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

/** argument struct for matrix-eltwise: opreduce vecs indexed */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meltw_opreduce_vecs_idx_param {
  unsigned long long n;
  const void* indices;       /* index array pointer */
  const void* in_matrix;     /* input matrix pointer */
  const void* in_vec;        /* input vector pointer */
  void* out_vec;             /* output pointer */
  const void* scale_vals;    /* scale values of indexed vectors after ops */
  const void* indices2;       /* index array pointer */
  const void* in_matrix2;     /* input matrix pointer */
  void* argop_off_vec_0;
  void* argop_off_vec_1;
} libxsmm_meltw_opreduce_vecs_idx_param;

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

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_matrix_arg_attributes {
  libxsmm_matrix_arg_type     type;
  libxsmm_matrix_arg_set_type set_type;
  libxsmm_blasint             set_cardinality_hint;
  libxsmm_blasint             set_stride_hint;
} libxsmm_matrix_arg_attributes;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_matrix_eqn_op_metadata {
  libxsmm_blasint eqn_idx;
  libxsmm_blasint op_arg_pos;
} libxsmm_matrix_eqn_op_metadata;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_matrix_eqn_arg_metadata {
  libxsmm_blasint eqn_idx;
  libxsmm_blasint in_arg_pos;
} libxsmm_matrix_eqn_arg_metadata;

/** argument struct for matrix-eltwise: unary */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meltw_unary_param {
  libxsmm_matrix_op_arg op;   /* op state & parameters */
  libxsmm_matrix_arg in;      /* input  */
  libxsmm_matrix_arg out;     /* output */
} libxsmm_meltw_unary_param;

/** argument struct for matrix-eltwise: binary */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meltw_binary_param {
  libxsmm_matrix_op_arg op;   /* op state & paramters */
  libxsmm_matrix_arg in0;     /* 1st input  */
  libxsmm_matrix_arg in1;     /* 2nd input  */
  libxsmm_matrix_arg out;     /* output     */
} libxsmm_meltw_binary_param;

/** argument struct for matrix-eltwise: ternary */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meltw_ternary_param {
  libxsmm_matrix_op_arg op;   /* op state & parameters */
  libxsmm_matrix_arg in0;     /* 1st input  */
  libxsmm_matrix_arg in1;     /* 2nd input  */
  libxsmm_matrix_arg in2;     /* 3rd input  */
  libxsmm_matrix_arg out;     /* output     */
} libxsmm_meltw_ternary_param;

/** argument struct for matrix equation */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_matrix_eqn_param {
  const libxsmm_matrix_op_arg* ops_args;    /* op state & parameters */
  const libxsmm_matrix_arg*    inputs;      /* array of input args */
  libxsmm_matrix_arg           output;      /* output arg */
} libxsmm_matrix_eqn_param;

/** Specialized function for matrix-eltw (weak-typed). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_meltwfunction_opreduce_vecs_idx)(const libxsmm_meltw_opreduce_vecs_idx_param* in_struct);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_meltwfunction_unary)(const libxsmm_meltw_unary_param* in_struct);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_meltwfunction_binary)(const libxsmm_meltw_binary_param* in_struct);
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_meltwfunction_ternary)(const libxsmm_meltw_ternary_param* in_struct);
/* matrix equation function */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_matrix_eqn_function)(const libxsmm_matrix_eqn_param* in_struct);

LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_xmeltwfunction {
  void (*xmeltw)(const void* in_struct);
  libxsmm_meltwfunction_opreduce_vecs_idx meltw_opreduce_vecs_idx;
  libxsmm_meltwfunction_unary meltw_unary;
  libxsmm_meltwfunction_binary meltw_binary;
  libxsmm_meltwfunction_ternary meltw_ternary;
} libxsmm_xmeltwfunction;

/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (double-precision). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_dmmfunction)(const double* a, const double* b, double* c);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (single-precision). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_smmfunction)(const float* a, const float* b, float* c);

/* argument structs for generalized interface */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_gemm_param {
  libxsmm_matrix_op_arg op;  /* op state & parameters */
  libxsmm_matrix_arg_v2 a;   /* a matrix  */
  libxsmm_matrix_arg_v2 b;   /* b matrix  */
  libxsmm_matrix_arg_v2 c;   /* c matrix  */
} libxsmm_gemm_param;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_gemm_ext_param {
  libxsmm_matrix_op_arg op;  /* op state & parameters */
  libxsmm_matrix_arg_v2 a;   /* a matrix  */
  libxsmm_matrix_arg_v2 b;   /* b matrix  */
  libxsmm_matrix_arg_v2 c;   /* c matrix  */
  libxsmm_matrix_arg_v2 d;   /* additional tensor for binary op on c */
  libxsmm_matrix_arg_v2 ap;  /* a after applying unary op */
  libxsmm_matrix_arg_v2 bp;  /* b after applying unary op */
  libxsmm_matrix_arg_v2 cp;  /* c before applying binary/ternary op after GEMM */
} libxsmm_gemm_ext_param;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_gemm_shape {
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

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_gemm_batch_reduce_config {
  libxsmm_gemm_batch_reduce_type br_type;  /* specifying the type of the BRGEMM operation */
  libxsmm_blasint br_stride_a_hint;        /* mandatory hint for strided BRGEMM */
  libxsmm_blasint br_stride_b_hint;        /* mandatory hint for strided BRGEMM */
  unsigned char br_unroll_hint;            /* optional hint containing the BR count */
} libxsmm_gemm_batch_reduce_config;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_gemm_ext_unary_argops {
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

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_gemm_ext_binary_postops {
  libxsmm_blasint ldd;                        /* leading dimensions of D */
  libxsmm_datatype d_in_type;                 /* datatype of D */
  libxsmm_meltw_binary_type d_binary_type;    /* op type for C = binaryry( C, D ) */
  libxsmm_bitfield d_binary_flags;            /* flags for C = binary( C, D ) */
} libxsmm_gemm_ext_binary_postops;

/* generalized and extended functions for everything that is not a basic GEMM as defined above */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_gemmfunction)    ( const libxsmm_gemm_param*     in_struct );
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE void (*libxsmm_gemmfunction_ext)( const libxsmm_gemm_ext_param* in_struct );

/** Function type which is either libxsmm_smmfunction or libxsmm_dmmfunction (weak-typed). */
LIBXSMM_EXTERN_C typedef union LIBXSMM_RETARGETABLE libxsmm_xmmfunction {
  void (*xmm)(const void* a, const void* b, void* c);
  void (*xgemm)(const void* in_struct);
  libxsmm_dmmfunction dmm; libxsmm_smmfunction smm;
  libxsmm_gemmfunction gemm; libxsmm_gemmfunction_ext gemm_ext;
} libxsmm_xmmfunction;

/** Structure to receive information about GEMM-kernels (libxsmm_get_mmkernel_info). */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_mmkernel_info {
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
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_meltwkernel_info {
  /** LDx, M, and N. */
  unsigned int ldi, ldo, m, n;
  /** Size of data element. */
  unsigned int datatype;
  /** Set of flags. */
  unsigned int flags;
  /** Set of operation. */
  unsigned int operation;
} libxsmm_meltwkernel_info;

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

