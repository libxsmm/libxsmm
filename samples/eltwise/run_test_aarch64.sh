#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

cd ${HERE} && cat <<EOM | ${EXEC}
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
./kernel_test/unary_copy_32b_eqld.sh
./kernel_test/unary_copy_32b_gtld.sh
./kernel_test/unary_xor_32b_eqld.sh
./kernel_test/unary_xor_32b_gtld.sh
./kernel_test/unary_sqrt_32b_eqld.sh
./kernel_test/unary_sqrt_32b_gtld.sh
./kernel_test/unary_x2_32b_eqld.sh
./kernel_test/unary_x2_32b_gtld.sh
./kernel_test/unary_negate_32b_eqld.sh
./kernel_test/unary_negate_32b_gtld.sh
./kernel_test/unary_exp_32b_eqld.sh
./kernel_test/unary_exp_32b_gtld.sh
./kernel_test/unary_inc_32b_eqld.sh
./kernel_test/unary_inc_32b_gtld.sh
./kernel_test/unary_rcp_32b_eqld.sh
./kernel_test/unary_rcp_32b_gtld.sh
./kernel_test/unary_rcp_sqrt_32b_eqld.sh
./kernel_test/unary_rcp_sqrt_32b_gtld.sh
./kernel_test/unary_relu_32b_eqld.sh
./kernel_test/unary_relu_32b_gtld.sh
./kernel_test/unary_dropout_32b_eqld.sh 33
./kernel_test/unary_dropout_32b_gtld.sh 33
./kernel_test/binary_add_32b_eqld.sh
./kernel_test/binary_add_32b_gtld.sh
./kernel_test/binary_mul_32b_eqld.sh
./kernel_test/binary_mul_32b_gtld.sh
./kernel_test/binary_sub_32b_eqld.sh
./kernel_test/binary_sub_32b_gtld.sh
./kernel_test/binary_div_32b_gtld.sh
./kernel_test/binary_div_32b_eqld.sh
./kernel_test/binary_muladd_32b_gtld.sh
./kernel_test/binary_muladd_32b_eqld.sh
./kernel_test/unary_gelu_32b_eqld.sh
./kernel_test/unary_gelu_32b_gtld.sh
./kernel_test/unary_gelu_inv_32b_eqld.sh
./kernel_test/unary_gelu_inv_32b_gtld.sh
./kernel_test/unary_tanh_32b_eqld.sh
./kernel_test/unary_tanh_32b_gtld.sh
./kernel_test/unary_tanh_inv_32b_eqld.sh
./kernel_test/unary_tanh_inv_32b_gtld.sh
./kernel_test/unary_sigmoid_32b_eqld.sh
./kernel_test/unary_sigmoid_32b_gtld.sh
./kernel_test/unary_sigmoid_inv_32b_eqld.sh
./kernel_test/unary_sigmoid_inv_32b_gtld.sh
EOM
RESULT=$?

rm -f tmp.??????????
exit ${RESULT}
