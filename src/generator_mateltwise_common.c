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
libxsmm_blasint libxsmm_generator_mateltwise_all_inp_comp_out_prec(const libxsmm_meltw_descriptor*   i_mateltwise_desc, libxsmm_datatype i_dtype ) {
  libxsmm_blasint result = 0;
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
    if (i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    if (i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) &&
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY )  {
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
  if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_UNARY ) {
    if (i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY ) {
    if (i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN0) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_IN1) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_COMP) ||
        i_dtype == (libxsmm_datatype) libxsmm_meltw_getenum_precision(i_mateltwise_desc, LIBXSMM_MELTW_FIELD_OUT)) {
      result = 1;
    }
  } else if (i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_TERNARY )  {
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
  if ((i_mateltwise_desc->operation == LIBXSMM_MELTW_OPERATION_BINARY) &&
      (i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GT ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_GE ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LT ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_LE ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_EQ ||
       i_mateltwise_desc->param == LIBXSMM_MELTW_TYPE_BINARY_CMP_OP_NE)) {
    result = 1;
  }
  return result;
}
