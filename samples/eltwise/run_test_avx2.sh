#!/usr/bin/env bash

export LIBXSMM_TARGET=hsw

cat <<EOM | ../../scripts/tool_pexec.sh
./kernel_test/binary_add_32b_eqld.sh
./kernel_test/binary_add_32b_gtld.sh
./kernel_test/binary_div_32b_eqld.sh
./kernel_test/binary_div_32b_gtld.sh
./kernel_test/binary_mul_32b_eqld.sh
./kernel_test/binary_mul_32b_gtld.sh
./kernel_test/binary_muladd_32b_eqld.sh
./kernel_test/binary_muladd_32b_gtld.sh
./kernel_test/binary_sub_32b_eqld.sh
./kernel_test/binary_sub_32b_gtld.sh
./kernel_test/unary_copy_32b_eqld.sh
./kernel_test/unary_copy_32b_gtld.sh
./kernel_test/unary_dropout_32b_eqld.sh 33
./kernel_test/unary_dropout_32b_gtld.sh 33
./kernel_test/unary_exp_32b_eqld.sh
./kernel_test/unary_exp_32b_gtld.sh
./kernel_test/unary_gelu_32b_eqld.sh
./kernel_test/unary_gelu_32b_gtld.sh
./kernel_test/unary_gelu_inv_32b_eqld.sh
./kernel_test/unary_gelu_inv_32b_gtld.sh
./kernel_test/unary_inc_32b_eqld.sh
./kernel_test/unary_inc_32b_gtld.sh
./kernel_test/unary_negate_32b_eqld.sh
./kernel_test/unary_negate_32b_gtld.sh
./kernel_test/unary_rcp_32b_eqld.sh
./kernel_test/unary_rcp_32b_gtld.sh
./kernel_test/unary_rcp_sqrt_32b_eqld.sh
./kernel_test/unary_rcp_sqrt_32b_gtld.sh
./kernel_test/unary_relu_32b_eqld.sh
./kernel_test/unary_relu_32b_gtld.sh
./kernel_test/unary_replicate_col_var_32b_eqld.sh
./kernel_test/unary_replicate_col_var_32b_gtld.sh
./kernel_test/unary_sigmoid_32b_eqld.sh
./kernel_test/unary_sigmoid_32b_gtld.sh
./kernel_test/unary_sigmoid_inv_32b_eqld.sh
./kernel_test/unary_sigmoid_inv_32b_gtld.sh
./kernel_test/unary_sqrt_32b_eqld.sh
./kernel_test/unary_sqrt_32b_gtld.sh
./kernel_test/unary_tanh_32b_eqld.sh
./kernel_test/unary_tanh_32b_gtld.sh
./kernel_test/unary_tanh_inv_32b_eqld.sh
./kernel_test/unary_tanh_inv_32b_gtld.sh
./kernel_test/unary_trans_08b_eqld.sh
./kernel_test/unary_trans_08b_gtld.sh
./kernel_test/unary_trans_16b_eqld.sh
./kernel_test/unary_trans_16b_gtld.sh
./kernel_test/unary_trans_32b_eqld.sh
./kernel_test/unary_trans_32b_gtld.sh
./kernel_test/unary_trans_64b_eqld.sh
./kernel_test/unary_trans_64b_gtld.sh
./kernel_test/unary_trans_padm_mod2_16b_eqld.sh
./kernel_test/unary_trans_padm_mod2_16b_gtld.sh
./kernel_test/unary_trans_padnm_mod2_16b_eqld.sh
./kernel_test/unary_trans_padnm_mod2_16b_gtld.sh
./kernel_test/unary_trans_padn_mod2_16b_eqld.sh
./kernel_test/unary_trans_padn_mod2_16b_gtld.sh
./kernel_test/unary_vnni_16b_eqld.sh
./kernel_test/unary_vnni_16b_gtld.sh
./kernel_test/unary_vnnitrans_16b_eqld.sh
./kernel_test/unary_vnnitrans_16b_gtld.sh
./kernel_test/unary_x2_32b_eqld.sh
./kernel_test/unary_x2_32b_gtld.sh
./kernel_test/unary_xor_32b_eqld.sh
./kernel_test/unary_xor_32b_gtld.sh
./kernel_test/reduce_add_cols_x2_32b_eqld.sh
./kernel_test/reduce_add_cols_x2_32b_gtld.sh
./kernel_test/reduce_add_cols_x_32b_eqld.sh
./kernel_test/reduce_add_cols_x_32b_gtld.sh
./kernel_test/reduce_add_cols_x_x2_32b_eqld.sh
./kernel_test/reduce_add_cols_x_x2_32b_gtld.sh
./kernel_test/reduce_add_idxcols_32b_eqld.sh
./kernel_test/reduce_add_idxcols_32b_gtld.sh
./kernel_test/reduce_add_rows_x2_32b_eqld.sh
./kernel_test/reduce_add_rows_x2_32b_gtld.sh
./kernel_test/reduce_add_rows_x_32b_eqld.sh
./kernel_test/reduce_add_rows_x_32b_gtld.sh
./kernel_test/reduce_add_rows_x_x2_32b_eqld.sh
./kernel_test/reduce_add_rows_x_x2_32b_gtld.sh
./kernel_test/reduce_max_cols_32b_eqld.sh
./kernel_test/reduce_max_cols_32b_gtld.sh
./kernel_test/reduce_max_rows_32b_eqld.sh
./kernel_test/reduce_max_rows_32b_gtld.sh
./kernel_test/unary_gather_16b_eqld.sh
./kernel_test/unary_gather_16b_gtld.sh
./kernel_test/unary_gather_32b_eqld.sh
./kernel_test/unary_gather_32b_gtld.sh
./kernel_test/unary_scatter_16b_eqld.sh
./kernel_test/unary_scatter_16b_gtld.sh
./kernel_test/unary_scatter_32b_eqld.sh
./kernel_test/unary_scatter_32b_gtld.sh
EOM

rm -f tmp.??????????
unset LIBXSMM_TARGET
