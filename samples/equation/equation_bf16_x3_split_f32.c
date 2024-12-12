/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kirill Voronin (Intel Corp.)
******************************************************************************/
#include <libxsmm_utils.h>
#include <libxsmm.h>

unsigned int is_reference_kernel = 0;
libxsmm_kernel_info info;

int main(int argc, char* argv[]) {
  int ret = EXIT_SUCCESS;
  double error_bound = 0.0000005;
  long M = 31;
  long N = 15;
  libxsmm_blasint ld = N + 5;
  long i;
  long j;
  libxsmm_matdiff_info norms;

  libxsmm_blasint ld_dump;

  libxsmm_meqn_arg_shape arg_shape_in, arg_shape_out;
  libxsmm_meqn_arg_metadata arg_metadata;
  libxsmm_meqn_op_metadata op_metadata;
  libxsmm_matrix_arg_attributes arg_singular_attr = libxsmm_create_matrix_arg_attributes( LIBXSMM_MATRIX_ARG_TYPE_SINGULAR, LIBXSMM_MATRIX_ARG_SET_TYPE_NONE, 0, 0);
  libxsmm_blasint my_eqn0;
  libxsmm_meqn_function func0;
  libxsmm_matrix_arg arg_array[2];
  libxsmm_matrix_op_arg op_arg_arr[2];
  libxsmm_meqn_param eqn_param;

  float *naive_input;
  float *naive_output;
  libxsmm_bfloat16 *naive_output0;
  libxsmm_bfloat16 *naive_output1;
  libxsmm_bfloat16 *naive_output2;
  float *output_libxsmm;
  libxsmm_bfloat16 *output0_libxsmm;
  libxsmm_bfloat16 *output1_libxsmm;
  libxsmm_bfloat16 *output2_libxsmm;

  if (argc > 1) M  = atoi(argv[1]);
  if (argc > 2) N  = atoi(argv[2]);
  if (argc > 3) ld = atoi(argv[3]);

  ld_dump = M;

  naive_input = (float*)libxsmm_aligned_malloc( sizeof(float)*N*ld, 2097152);
  naive_output = (float*)libxsmm_aligned_malloc( sizeof(float)*N*ld, 2097152);
  naive_output0 = (libxsmm_bfloat16*)libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld, 2097152);
  naive_output1 = (libxsmm_bfloat16*)libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld, 2097152);
  naive_output2 = (libxsmm_bfloat16*)libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld, 2097152);
  output_libxsmm = (float*)libxsmm_aligned_malloc( sizeof(float)*N*ld, 2097152);
  output0_libxsmm = (libxsmm_bfloat16*)libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld_dump, 2097152);
  output1_libxsmm = (libxsmm_bfloat16*)libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld_dump, 2097152);
  output2_libxsmm = (libxsmm_bfloat16*)libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*ld, 2097152);

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < ld; ++j ) {
      naive_input[i*ld + j] = (float)libxsmm_rng_f64();
    }
  }

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float ftmp = naive_input[i*ld + j];
      float ftmp0, ftmp1, ftmp2;

      libxsmm_rne_convert_fp32_bf16(&ftmp, &naive_output0[i*ld + j], 1);
      libxsmm_convert_bf16_f32(&naive_output0[i*ld + j], &ftmp0, 1);
      ftmp -= ftmp0;

      libxsmm_rne_convert_fp32_bf16(&ftmp, &naive_output1[i*ld + j], 1);
      libxsmm_convert_bf16_f32(&naive_output1[i*ld + j], &ftmp1, 1);
      ftmp -= ftmp1;

      libxsmm_rne_convert_fp32_bf16(&ftmp, &naive_output2[i*ld + j], 1);
      libxsmm_convert_bf16_f32(&naive_output2[i*ld + j], &ftmp2, 1);
      ftmp -= ftmp2;
    }
  }

  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float ftmp0, ftmp1, ftmp2;
      libxsmm_convert_bf16_f32(&naive_output0[i*ld + j], &ftmp0, 1);
      libxsmm_convert_bf16_f32(&naive_output1[i*ld + j], &ftmp1, 1);
      libxsmm_convert_bf16_f32(&naive_output2[i*ld + j], &ftmp2, 1);

      naive_output[i*ld + j] = ftmp0 + ftmp1 + ftmp2;
    }
  }

  my_eqn0 = libxsmm_meqn_create();
  op_metadata = libxsmm_create_meqn_op_metadata( my_eqn0, -1 );
  libxsmm_meqn_push_back_unary_op( op_metadata, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE ); /* not sure about dtype */
  libxsmm_meqn_push_back_binary_op( op_metadata, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE );
  libxsmm_meqn_push_back_binary_op( op_metadata, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE );
  arg_shape_in  = libxsmm_create_meqn_arg_shape( M, N, ld, LIBXSMM_DATATYPE_F32 );
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 0);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  libxsmm_meqn_push_back_unary_op( op_metadata, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE ); /* not sure about dtype */
  arg_shape_in  = libxsmm_create_meqn_arg_shape( M, N, ld_dump, LIBXSMM_DATATYPE_BF16 );
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 1);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  libxsmm_meqn_push_back_unary_op( op_metadata, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE ); /* not sure about dtype */
  op_metadata.op_arg_pos   = 1;
  libxsmm_meqn_push_back_unary_op( op_metadata, LIBXSMM_MELTW_TYPE_UNARY_DUMP, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE ); /* b1 */
  op_metadata.op_arg_pos   = -1;
  libxsmm_meqn_push_back_unary_op( op_metadata, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  libxsmm_meqn_push_back_binary_op( op_metadata, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE );
  arg_shape_in  = libxsmm_create_meqn_arg_shape( M, N, ld, LIBXSMM_DATATYPE_F32 );
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 0);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  libxsmm_meqn_push_back_unary_op( op_metadata, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE ); /* not sure about dtype */
  op_metadata.op_arg_pos   = 0;
  libxsmm_meqn_push_back_unary_op( op_metadata, LIBXSMM_MELTW_TYPE_UNARY_DUMP, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE ); /* b0 */
  op_metadata.op_arg_pos   = -1;
  libxsmm_meqn_push_back_unary_op( op_metadata, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  arg_shape_in  = libxsmm_create_meqn_arg_shape( M, N, ld, LIBXSMM_DATATYPE_F32 ); /* input a */
  arg_metadata  = libxsmm_create_meqn_arg_metadata(my_eqn0, 0);
  libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
  arg_shape_out = libxsmm_create_meqn_arg_shape( M, N, ld, LIBXSMM_DATATYPE_BF16 );

  /* libxsmm_meqn_tree_print(my_eqn0); */
  /* libxsmm_meqn_rpn_print(my_eqn0); */
  func0 = libxsmm_dispatch_meqn( my_eqn0, arg_shape_out );
  libxsmm_get_kernel_info((const void*) func0, &info);
  is_reference_kernel = info.is_reference_kernel;
  if ( func0 == NULL ) {
    fprintf( stderr, "JIT for func0 failed. Bailing...!\n");
    exit(-1);
  }

  memset( &eqn_param, 0, sizeof(eqn_param));
  eqn_param.inputs   = arg_array;
  eqn_param.ops_args = op_arg_arr;

  arg_array[0].primary     = (void*)naive_input;
  arg_array[1].primary     = (void*)output0_libxsmm;
  op_arg_arr[0].primary    = (void*)output0_libxsmm;
  op_arg_arr[1].primary    = (void*)output1_libxsmm;
  eqn_param.output.primary = (void*)output2_libxsmm;

  func0(&eqn_param);

  /* Note that the dumped buffers have ld_dump = M per how DUMP operator works */
  for ( i = 0; i < N; ++i ) {
    for ( j = 0; j < M; ++j ) {
      float ftmp0, ftmp1, ftmp2;
      libxsmm_convert_bf16_f32(&output0_libxsmm[i*ld_dump + j], &ftmp0, 1);
      libxsmm_convert_bf16_f32(&output1_libxsmm[i*ld_dump + j], &ftmp1, 1);
      libxsmm_convert_bf16_f32(&output2_libxsmm[i*ld + j], &ftmp2, 1);

      output_libxsmm[i*ld + j] = ftmp0 + ftmp1 + ftmp2;
    }
  }

  libxsmm_matdiff_clear(&norms);

  printf("##########################################\n");
  printf("#   Correctness [naive vs naive split]   #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, M, N, naive_input, naive_output, &ld, &ld);
  printf("L1 reference  : %.25g\n", norms.l1_ref);
  printf("L1 test       : %.25g\n", norms.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms.l2_rel);
  printf("Linf abs.error: %.24f\n", norms.linf_abs);
  printf("Linf rel.error: %.24f\n", norms.linf_rel);
  printf("Check-norm    : %.24f\n", norms.normf_rel);

  if ( norms.normf_rel > error_bound ) {
    ret = EXIT_FAILURE;
  }

  printf("##########################################\n");
  printf("#   Correctness [naive vs libxsmm split] #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_F32, M, N, naive_output, output_libxsmm, &ld, &ld);
  printf("L1 reference  : %.25g\n", norms.l1_ref);
  printf("L1 test       : %.25g\n", norms.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms.l2_rel);
  printf("Linf abs.error: %.24f\n", norms.linf_abs);
  printf("Linf rel.error: %.24f\n", norms.linf_rel);
  printf("Check-norm    : %.24f\n", norms.normf_rel);

  if ( norms.normf_rel > error_bound ) {
    ret = EXIT_FAILURE;
  }

  printf("##########################################\n");
  printf("# Correctness [naive0 vs libxsmm0 split] #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_BF16, M, N, naive_output0, output0_libxsmm, &ld, &ld_dump);
  printf("L1 reference  : %.25g\n", norms.l1_ref);
  printf("L1 test       : %.25g\n", norms.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms.l2_rel);
  printf("Linf abs.error: %.24f\n", norms.linf_abs);
  printf("Linf rel.error: %.24f\n", norms.linf_rel);
  printf("Check-norm    : %.24f\n", norms.normf_rel);

  if ( norms.normf_rel > error_bound ) {
    ret = EXIT_FAILURE;
  }

  printf("##########################################\n");
  printf("# Correctness [naive1 vs libxsmm1 split] #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_BF16, M, N, naive_output1, output1_libxsmm, &ld, &ld_dump);
  printf("L1 reference  : %.25g\n", norms.l1_ref);
  printf("L1 test       : %.25g\n", norms.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms.l2_rel);
  printf("Linf abs.error: %.24f\n", norms.linf_abs);
  printf("Linf rel.error: %.24f\n", norms.linf_rel);
  printf("Check-norm    : %.24f\n", norms.normf_rel);

  if ( norms.normf_rel > error_bound ) {
    ret = EXIT_FAILURE;
  }

  printf("##########################################\n");
  printf("# Correctness [naive2 vs libxsmm2 split] #\n");
  printf("##########################################\n");
  libxsmm_matdiff(&norms, LIBXSMM_DATATYPE_BF16, M, N, naive_output2, output2_libxsmm, &ld, &ld);
  printf("L1 reference  : %.25g\n", norms.l1_ref);
  printf("L1 test       : %.25g\n", norms.l1_tst);
  printf("L2 abs.error  : %.24f\n", norms.l2_abs);
  printf("L2 rel.error  : %.24f\n", norms.l2_rel);
  printf("Linf abs.error: %.24f\n", norms.linf_abs);
  printf("Linf rel.error: %.24f\n", norms.linf_rel);
  printf("Check-norm    : %.24f\n", norms.normf_rel);

  if ( norms.normf_rel > error_bound ) {
    ret = EXIT_FAILURE;
  }

  if (ret != EXIT_SUCCESS) {
    printf("FAILED\n");
  } else {
    printf("SUCCESS\n");
  }

  libxsmm_free(naive_input);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output0);
  libxsmm_free(naive_output1);
  libxsmm_free(naive_output2);
  libxsmm_free(output_libxsmm);
  libxsmm_free(output0_libxsmm);
  libxsmm_free(output1_libxsmm);
  libxsmm_free(output2_libxsmm);

  ret = (ret == EXIT_SUCCESS) ? libxsmm_return_success_code(is_reference_kernel) : ret;
  return ret;
}
