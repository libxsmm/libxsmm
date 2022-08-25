#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

cd ${HERE} && cat <<EOM | ${EXEC} -c 3- "$@"
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
./kernel_test/unary_copy_eqld.sh
./kernel_test/unary_copy_gtld.sh
./kernel_test/unary_xor_eqld.sh
./kernel_test/unary_xor_gtld.sh
./kernel_test/unary_sqrt_eqld.sh
./kernel_test/unary_sqrt_gtld.sh
./kernel_test/unary_x2_eqld.sh
./kernel_test/unary_x2_gtld.sh
./kernel_test/unary_negate_eqld.sh
./kernel_test/unary_negate_gtld.sh
./kernel_test/unary_exp_eqld.sh
./kernel_test/unary_exp_gtld.sh
./kernel_test/unary_inc_eqld.sh
./kernel_test/unary_inc_gtld.sh
./kernel_test/unary_rcp_eqld.sh
./kernel_test/unary_rcp_gtld.sh
./kernel_test/unary_rcp_sqrt_eqld.sh
./kernel_test/unary_rcp_sqrt_gtld.sh
./kernel_test/unary_relu_eqld.sh
./kernel_test/unary_relu_gtld.sh
./kernel_test/unary_dropout_eqld.sh 33
./kernel_test/unary_dropout_gtld.sh 33
./kernel_test/binary_add_eqld.sh
./kernel_test/binary_add_gtld.sh
./kernel_test/binary_mul_eqld.sh
./kernel_test/binary_mul_gtld.sh
./kernel_test/binary_sub_eqld.sh
./kernel_test/binary_sub_gtld.sh
./kernel_test/binary_div_gtld.sh
./kernel_test/binary_div_eqld.sh
./kernel_test/binary_muladd_gtld.sh
./kernel_test/binary_muladd_eqld.sh
./kernel_test/unary_gelu_eqld.sh
./kernel_test/unary_gelu_gtld.sh
./kernel_test/unary_gelu_inv_eqld.sh
./kernel_test/unary_gelu_inv_gtld.sh
./kernel_test/unary_tanh_eqld.sh
./kernel_test/unary_tanh_gtld.sh
./kernel_test/unary_tanh_inv_eqld.sh
./kernel_test/unary_tanh_inv_gtld.sh
./kernel_test/unary_sigmoid_eqld.sh
./kernel_test/unary_sigmoid_gtld.sh
./kernel_test/unary_sigmoid_inv_eqld.sh
./kernel_test/unary_sigmoid_inv_gtld.sh
EOM
RESULT=$?

rm -f tmp.??????????
exit ${RESULT}
