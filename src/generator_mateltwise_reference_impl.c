/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "generator_mateltwise_common.h"
#include "generator_common.h"
#include "generator_mateltwise_reference_impl.h"
#include "utils/libxsmm_lpflt_quant.h"
#include "utils/libxsmm_math.h"

LIBXSMM_INLINE
float libxsmm_fsigmoid(float x) {
  return (LIBXSMM_TANHF(x/2.0f) + 1.0f)/2.0f;
}

LIBXSMM_INLINE
float libxsmm_fsigmoid_inv(float x) {
  return libxsmm_fsigmoid(x) * (1.0f-libxsmm_fsigmoid(x));
}

LIBXSMM_INLINE
float libxsmm_tanh_inv(float x) {
  return 1.0f-LIBXSMM_TANHF(x)*LIBXSMM_TANHF(x);
}

LIBXSMM_INLINE
float libxsmm_gelu(float x) {
  return (LIBXSMM_ERFF(x/LIBXSMM_SQRTF(2.0f)) + 1.0f)*0.5f*x;
}

LIBXSMM_INLINE
float libxsmm_gelu_inv(float x) {
  return (0.5f + 0.5f * LIBXSMM_ERFF(x/LIBXSMM_SQRTF(2.0f)) + x/(LIBXSMM_SQRTF(2.0f*M_PI)) * LIBXSMM_EXPF(-0.5f*x*x) );
}

LIBXSMM_INLINE
float libxsmm_fp32_unary_compute(float in, libxsmm_meltw_unary_type op) {
  float res = 0;
  if ( op == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY || op == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) {
    res = in;
  } else if ( op == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
    res = -1.0f * in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_X2) {
    res = in * in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_XOR) {
    res = 0;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_TANH) {
    res = LIBXSMM_TANHF(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID) {
    res = libxsmm_fsigmoid(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU) {
    res = libxsmm_gelu(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_GELU_INV) {
    res = libxsmm_gelu_inv(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_TANH_INV) {
    res = libxsmm_tanh_inv(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_SIGMOID_INV) {
    res = libxsmm_fsigmoid_inv(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
    res = LIBXSMM_SQRTF(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_INC) {
    res = in + 1.0f;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {
    res = 1.0f/in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {
    res = 1.0f/LIBXSMM_SQRTF(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_EXP) {
    res = LIBXSMM_EXPF(in);
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }
  return res;
}

LIBXSMM_INLINE
double libxsmm_fp64_unary_compute(double in, libxsmm_meltw_unary_type op) {
  double res = 0;
  if ( op == LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) {
    res = in;
  } else if ( op == LIBXSMM_MELTW_TYPE_UNARY_NEGATE) {
    res = -1.0 * in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_X2) {
    res = in * in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_XOR) {
    res = 0;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_SQRT) {
    res = sqrt(in);
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_INC) {
    res = in + 1.0;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL) {
    res = 1.0/in;
  } else if (op == LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL_SQRT) {
    res = 1.0/sqrt(in);
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }
  return res;
}

LIBXSMM_INLINE
int libxsmm_is_cmp_op(libxsmm_meltw_binary_type op) {
  int result = 0;
  if (op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GT || op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GE || op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT || op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE || op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_EQ || op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE) {
    result = 1;
  }
  return result;
}

LIBXSMM_INLINE
void libxsmm_set_bit(unsigned char *bit_matrix, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld) {
  libxsmm_blasint byte_pos = i/8;
  libxsmm_blasint pos_in_byte = i%8;
  unsigned char byte_to_write = 1;
  byte_to_write = (unsigned char)(byte_to_write << pos_in_byte);
  bit_matrix[byte_pos + j * (ld/8)] |= byte_to_write;
}

LIBXSMM_INLINE
void libxsmm_zero_bit(unsigned char *bit_matrix, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld) {
  libxsmm_blasint byte_pos = i/8;
  libxsmm_blasint pos_in_byte = i%8;
  unsigned char and_mask = 1;
  and_mask = (unsigned char)(and_mask << pos_in_byte);
  and_mask = ~and_mask;
  bit_matrix[byte_pos + j * (ld/8)] = bit_matrix[byte_pos + j * (ld/8)] & and_mask;
}

LIBXSMM_INLINE
float libxsmm_fp32_binary_compute(float in0, float in1, float out, libxsmm_meltw_binary_type op) {
  float res = out;
  if ( op == LIBXSMM_MELTW_TYPE_BINARY_ADD) {
    res = in0 + in1;
  } else  if ( op == LIBXSMM_MELTW_TYPE_BINARY_SUB) {
    res = in0 - in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MUL) {
    res = in0 * in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_DIV) {
    res = in0 / in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
    res += in0 * in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MAX) {
    res = (in0 > in1) ? in0 : in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MIN) {
    res = (in0 > in1) ? in1 : in0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GT) {
    res = (in0 > in1) ? 1 : 0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GE) {
    res = (in0 >= in1 ) ? 1 : 0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT) {
    res = (in0 < in1) ? 1 : 0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE) {
    res = (in0 <= in1) ? 1 : 0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_EQ) {
    res = (in0 == in1) ? 1 : 0;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE) {
    res = (in0 != in1) ? 1 : 0;
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }
  return res;
}

LIBXSMM_INLINE
double libxsmm_fp64_binary_compute(double in0, double in1, double out, libxsmm_meltw_binary_type op) {
  double res = out;
  if ( op == LIBXSMM_MELTW_TYPE_BINARY_ADD) {
    res = in0 + in1;
  } else  if ( op == LIBXSMM_MELTW_TYPE_BINARY_SUB) {
    res = in0 - in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MUL) {
    res = in0 * in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_DIV) {
    res = in0 / in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MULADD) {
    res += in0 * in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MAX) {
    res = (in0 > in1) ? in0 : in1;
  } else if ( op == LIBXSMM_MELTW_TYPE_BINARY_MIN) {
    res = (in0 > in1) ? in1 : in0;
  } else {
    printf("Invalid OP\n");
    exit(-1);
  }
  return res;
}

LIBXSMM_INLINE
libxsmm_blasint libxsmm_elementwise_get_index(libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld, const libxsmm_meltw_descriptor *i_mateltwise_desc, unsigned int operand_id) {
  libxsmm_blasint result = 0;
  unsigned int bcast_row = (((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_0) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_1) > 0)) ||
                            ((operand_id == 2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_ROW_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_col = (((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_0) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_1) > 0)) ||
                            ((operand_id == 2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_COL_IN_2) > 0))) ? 1 : 0;
  unsigned int bcast_sca = (((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0) > 0)) ||
                            ((operand_id == 0) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1) > 0)) ||
                            ((operand_id == 1) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1) > 0)) ||
                            ((operand_id == 2) && (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) && ((i_mateltwise_desc->flags & LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2) > 0))) ? 1 : 0;
  if (bcast_row > 0) {
    result = j * ld;
  } else if (bcast_col > 0) {
    result = i;
  } else if (bcast_sca > 0) {
    result = 0;
  } else {
    result = i + j * ld;
  }
  return result;
}

LIBXSMM_INLINE
float libxsmm_elementwise_get_float_value(const void *in, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld, libxsmm_datatype dtype_in, const libxsmm_meltw_descriptor *i_mateltwise_desc, unsigned int operand_id) {
  float result = 0.0;
  if ( dtype_in == LIBXSMM_DATATYPE_F32 ) {
    const float* f_in = (const float*)in;
    result = f_in[libxsmm_elementwise_get_index(i, j, ld, i_mateltwise_desc, operand_id)];
  } else if ( dtype_in == LIBXSMM_DATATYPE_BF16 ) {
    const libxsmm_bfloat16* bf16_in = (const libxsmm_bfloat16*)in;
    result = libxsmm_convert_bf16_to_f32(bf16_in[libxsmm_elementwise_get_index(i, j, ld, i_mateltwise_desc, operand_id)]);
  } else if ( dtype_in == LIBXSMM_DATATYPE_F16 ) {
    const libxsmm_float16* f16_in = (const libxsmm_float16*)in;
    result = libxsmm_convert_f16_to_f32(f16_in[libxsmm_elementwise_get_index(i, j, ld, i_mateltwise_desc, operand_id)]);
  } else if ( dtype_in == LIBXSMM_DATATYPE_BF8 ) {
    const libxsmm_bfloat8* bf8_in = (const libxsmm_bfloat8*)in;
    result = libxsmm_convert_bf8_to_f32(bf8_in[libxsmm_elementwise_get_index(i, j, ld, i_mateltwise_desc, operand_id)]);
  } else if ( dtype_in == LIBXSMM_DATATYPE_HF8 ) {
    const libxsmm_hfloat8* hf8_in = (const libxsmm_hfloat8*)in;
    result = libxsmm_convert_hf8_to_f32(hf8_in[libxsmm_elementwise_get_index(i, j, ld, i_mateltwise_desc, operand_id)]);
  } else {
    /* should not happen */
  }
  return result;
}

LIBXSMM_INLINE
void libxsmm_elementwise_store_value(void *out, void* out_value_ptr, libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ldo, libxsmm_bitfield flags, libxsmm_datatype dtype_out, void *rng_state, unsigned int seed_idx) {
  float out_value = *((float*)out_value_ptr);
  if ( dtype_out == LIBXSMM_DATATYPE_F32 ) {
    float* f_out = (float*)out;
    f_out[(j*ldo) + i] = out_value;
  } else if ( dtype_out == LIBXSMM_DATATYPE_BF16 ) {
    libxsmm_bfloat16* bf16_out = (libxsmm_bfloat16*)out;
    bf16_out[(j*ldo) + i] = libxsmm_convert_f32_to_bf16_rne(out_value);
  } else if ( dtype_out == LIBXSMM_DATATYPE_F16 ) {
    libxsmm_float16* f16_out = (libxsmm_float16*)out;
    f16_out[(j*ldo) + i] = libxsmm_convert_f32_to_f16(out_value);
  } else if ( dtype_out == LIBXSMM_DATATYPE_BF8 ) {
    libxsmm_bfloat8* bf8_out = (libxsmm_bfloat8*)out;
    if ((flags & LIBXSMM_MELTW_FLAG_UNARY_STOCHASTIC_ROUND) > 0 ) {
      libxsmm_stochastic_convert_fp32_bf8(&out_value, &(bf8_out[(j*ldo) + i]), 1, rng_state, seed_idx);
    } else {
      bf8_out[(j*ldo) + i] = libxsmm_convert_f32_to_bf8_rne(out_value);
    }
  } else if ( dtype_out == LIBXSMM_DATATYPE_HF8 ) {
    libxsmm_hfloat8* hf8_out = (libxsmm_hfloat8*)out;
    hf8_out[(j*ldo) + i] = libxsmm_convert_f32_to_hf8_rne(out_value);
  } else {
    /* should not happen */
  }
  return;
}

LIBXSMM_INLINE
void libxsmm_reference_unary_elementwise(libxsmm_meltw_unary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  libxsmm_blasint M = i_mateltwise_desc->m;
  libxsmm_blasint N = (libxsmm_blasint) (( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) ? *((unsigned long long*)(param->op.primary)) : i_mateltwise_desc->n);
  libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_datatype dtype_in = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);
  libxsmm_datatype dtype_out = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT);
  libxsmm_datatype dtype_comp = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP);
  libxsmm_bitfield flags =  i_mateltwise_desc->flags;
  void *rng_state = (void*) param->op.secondary;
  void *in = (void*)param->in.primary;
  void *out = (void*)param->out.primary;
  unsigned int seed_idx = 0;

  if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_UNZIP ) {
    /* Special case for unzip TPP */
    float *_in = (float*)in;
    unsigned long long offset = *((unsigned long long*)(param->out.secondary));
    libxsmm_bfloat16 *out_lo = (libxsmm_bfloat16*)out;
    libxsmm_bfloat16 *out_hi = (libxsmm_bfloat16*)((char*)out + offset);
    libxsmm_blasint i, j;
    for (j = 0; j < N; j++) {
      for (i = 0; i < M; i++) {
        libxsmm_bfloat16_f32 bf16_hp;
        bf16_hp.f = _in[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
        out_lo[j * ldo + i] = bf16_hp.i[0];
        out_hi[j * ldo + i] = bf16_hp.i[1];
      }
    }
  } else if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X2 ||
             i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3) {
    unsigned long long *strides = (unsigned long long*)(param->out.secondary);
    /* Special case for decompose TPP */
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        float in_value = 0.0f;
        libxsmm_bfloat16 out1_value = 0, out2_value = 0, out3_value = 0;
        float ftmp = 0.0f, ftmp2 = 0.0f;
        libxsmm_bfloat16_f32 tmp;
        const float* f_in = (const float*)in;
        libxsmm_bfloat16* bf16_out = (libxsmm_bfloat16*)out;
        in_value = f_in[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
        tmp.f = in_value;
        tmp.i[0] = 0;
        out1_value = tmp.i[1];
        ftmp = in_value - tmp.f;
        if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3 ) {
          tmp.f = ftmp;
          tmp.i[0] = 0;
          out2_value = tmp.i[1];
          ftmp2 = ftmp - tmp.f;
          libxsmm_rne_convert_fp32_bf16(&ftmp2, &out3_value, 1);
        } else {
          libxsmm_rne_convert_fp32_bf16(&ftmp,  &out2_value, 1);
        }
        bf16_out[(j*ldo) + i                   ] = out1_value;
        bf16_out[(j*ldo) + i + (strides[0]/2)  ] = out2_value;
        if (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_UNARY_DECOMP_FP32_TO_BF16X3) {
          bf16_out[(j*ldo) + i + (strides[1]/2)] = out3_value;
        }
      }
    }
  } else {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        if (dtype_in == LIBXSMM_DATATYPE_F64 && dtype_out == LIBXSMM_DATATYPE_F64 && dtype_comp == LIBXSMM_DATATYPE_F64) {
          double *in_double = (double*)in;
          double *out_double = (double*)out;
          double in_val_double = in_double[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
          out_double[i + j * ldo] = libxsmm_fp64_unary_compute(in_val_double, i_mateltwise_desc->param);
        } else {
          float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
          float out_val = libxsmm_fp32_unary_compute(in_val, i_mateltwise_desc->param);
          libxsmm_elementwise_store_value(out, (void*)&out_val, i, j, ldo, flags, dtype_out, rng_state, seed_idx);
          seed_idx++;
        }
      }
    }
  }
  return;
}

LIBXSMM_INLINE
void libxsmm_reference_binary_elementwise(libxsmm_meltw_binary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  libxsmm_blasint i, j;
  libxsmm_blasint M = i_mateltwise_desc->m;
  libxsmm_blasint N = i_mateltwise_desc->n;
  libxsmm_blasint ldi = i_mateltwise_desc->ldi;
  libxsmm_blasint ldi1 = i_mateltwise_desc->ldi2;
  libxsmm_blasint ldo = i_mateltwise_desc->ldo;
  libxsmm_datatype dtype_in = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0);
  libxsmm_datatype dtype_in1 = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1);
  libxsmm_datatype dtype_out = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT);
  libxsmm_datatype dtype_comp = libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP);
  libxsmm_bitfield flags =  i_mateltwise_desc->flags;
  void *rng_state = (void*) param->op.secondary;
  void *in = (void*)param->in0.primary;
  void *in1 = (void*)param->in1.primary;
  void *out = (void*)param->out.primary;
  unsigned int seed_idx = 0;

  if ( i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_ZIP ) {
    /* Special case for zip TPP */
    float *out_res = (float*)out;
    libxsmm_bfloat16 *in_lo = (libxsmm_bfloat16*)in;
    libxsmm_bfloat16 *in_hi = (libxsmm_bfloat16*)in1;
    for (j = 0; j < N; j++) {
      for (i = 0; i < M; i++) {
        libxsmm_bfloat16_f32 bf16_hp;
        bf16_hp.i[0] = in_lo[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
        bf16_hp.i[1] = in_hi[libxsmm_elementwise_get_index(i, j, ldi1, i_mateltwise_desc, 1)];
        out_res[j * ldo + i] = bf16_hp.f;
      }
    }
  } else {
    for ( j = 0; j < N; ++j ) {
      for ( i = 0; i < M; ++i ) {
        if (dtype_in == LIBXSMM_DATATYPE_F64 && dtype_in1 == LIBXSMM_DATATYPE_F64 && dtype_out == LIBXSMM_DATATYPE_F64 && dtype_comp == LIBXSMM_DATATYPE_F64) {
          double *in_double = (double*)in;
          double *in1_double = (double*)in1;
          double *out_double = (double*)out;
          double in_val_double = in_double[libxsmm_elementwise_get_index(i, j, ldi, i_mateltwise_desc, 0)];
          double in1_val_double = in1_double[libxsmm_elementwise_get_index(i, j, ldi1, i_mateltwise_desc, 1)];
          out_double[i + j * ldo] = libxsmm_fp64_binary_compute(in_val_double, in1_val_double, out_double[i + j * ldo], i_mateltwise_desc->param);
        } else {
          float in_val  = libxsmm_elementwise_get_float_value(in, i, j, ldi, dtype_in, i_mateltwise_desc, 0);
          float in1_val  = libxsmm_elementwise_get_float_value(in1, i, j, ldi1, dtype_in1, i_mateltwise_desc, 1);
          if (libxsmm_is_cmp_op(i_mateltwise_desc->param) > 0) {
            unsigned int l_ldo = LIBXSMM_UPDIV(ldo, 16)*16;
            float out_value = libxsmm_fp32_binary_compute(in_val, in1_val, 0.0, i_mateltwise_desc->param);
            unsigned char result_bit = (out_value > 0.1) ? 1 : 0;
            if (result_bit > 0) {
              libxsmm_set_bit((unsigned char*)out, i, j, l_ldo);
            } else {
              libxsmm_zero_bit((unsigned char*)out, i, j, l_ldo);
            }
          } else {
            float out_in  = libxsmm_elementwise_get_float_value(out, i, j, ldo, dtype_out, i_mateltwise_desc, 3);
            float out_val = libxsmm_fp32_binary_compute(in_val, in1_val, out_in, i_mateltwise_desc->param);
            libxsmm_elementwise_store_value(out, (void*)&out_val, i, j, ldo, flags, dtype_out, rng_state, seed_idx);
          }
          seed_idx++;
        }
      }
    }
  }
  return;
}

LIBXSMM_INLINE
void libxsmm_reference_ternary_elementwise(libxsmm_meltw_ternary_param *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  return;
}

void libxsmm_reference_elementwise(void *param, const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY) {
    libxsmm_reference_unary_elementwise((libxsmm_meltw_unary_param*)param, i_mateltwise_desc);
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) {
    libxsmm_reference_binary_elementwise((libxsmm_meltw_binary_param*)param, i_mateltwise_desc);
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY) {
    libxsmm_reference_ternary_elementwise((libxsmm_meltw_ternary_param*)param, i_mateltwise_desc);
  } else {
    /* Should not happen  */
  }
  return;
}
