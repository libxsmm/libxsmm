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


LIBXSMM_API_INTERN
unsigned int libxsmm_is_transform_op(const libxsmm_meltw_descriptor *i_mateltwise_desc) {
  unsigned int result = 0;
  if ( libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI4T ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_VNNI8T ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2 ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2T ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8_TO_NORM ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI8T_TO_NORM ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4T_TO_NORM ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2T_TO_NORM ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4T ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8T ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4_PAD ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI4 ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8_PAD ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI8 ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_NORM ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI4_TO_VNNI2 ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD2 ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD2 ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD2 ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADN_MOD4 ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADM_MOD4 ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_PADNM_MOD4 ) {
    result = 1;
  }
  return result;
}

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_mateltwise_all_inp_comp_out_prec(const libxsmm_meltw_descriptor*   i_mateltwise_desc, libxsmm_datatype i_dtype ) {
  libxsmm_blasint result = 0;
  if (libxsmm_meltw_descriptor_get_operation(i_mateltwise_desc) == LIBXSMM_MELTW_OPERATION_UNARY ) {
    if (i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (libxsmm_meltw_descriptor_get_operation(i_mateltwise_desc) == LIBXSMM_MELTW_OPERATION_BINARY ) {
    if (i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (libxsmm_meltw_descriptor_get_operation(i_mateltwise_desc) == LIBXSMM_MELTW_OPERATION_TERNARY )  {
    if (i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else {
    /* Should not happen */
  }
  return result;
}

LIBXSMM_API_INTERN
libxsmm_blasint libxsmm_generator_mateltwise_involves_prec(const libxsmm_meltw_descriptor*   i_mateltwise_desc, libxsmm_datatype i_dtype ) {
  libxsmm_blasint result = 0;
  if (libxsmm_meltw_descriptor_get_operation(i_mateltwise_desc) == LIBXSMM_MELTW_OPERATION_UNARY ) {
    if (i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (libxsmm_meltw_descriptor_get_operation(i_mateltwise_desc) == LIBXSMM_MELTW_OPERATION_BINARY ) {
    if (i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (libxsmm_meltw_descriptor_get_operation(i_mateltwise_desc) == LIBXSMM_MELTW_OPERATION_TERNARY )  {
    if (i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN2) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else {
    /* Should not happen */
  }
  return result;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_generator_mateltwise_is_binary_cmp_op( const libxsmm_meltw_descriptor*         i_mateltwise_desc) {
  unsigned int result = 0;
  if ((libxsmm_meltw_descriptor_get_operation(i_mateltwise_desc) == LIBXSMM_MELTW_OPERATION_BINARY) &&
      (libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GT ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GE ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_EQ ||
       libxsmm_meltw_descriptor_get_param(i_mateltwise_desc) == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE)) {
    result = 1;
  }
  return result;
}
