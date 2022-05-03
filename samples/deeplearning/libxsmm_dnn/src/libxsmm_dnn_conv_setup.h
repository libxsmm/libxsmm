/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

#ifndef LIBXSMM_DNN_CONV_SETUP_H
#define LIBXSMM_DNN_CONV_SETUP_H

#include <libxsmm_dnn_conv.h>

LIBXSMM_API_INLINE void  libxsmm_dnn_conv_get_feature_map_blocks( int C, int K, int* C_block, int* K_block, int* fm_lp_block, libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_blasint bc, libxsmm_blasint bk );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_ifmblock( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_ofmblock( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fm_lp_block( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fallback_loops_fwd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksifm( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksofm( libxsmm_dnn_conv_config* cfg );

/**********************************************************/
/* Helper functions for FWD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_ofw_rb( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_pack_input_fwd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_ofh_rb( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_pixels_gemm( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_block_H( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksifm_blocking( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_loop_order_fwd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_fwd_IFM( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_fwd_OFM( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_use_ofm_parallelization( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_rim_fmas_fwd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_shuffle_filter_accesses( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_acc_load( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_init_fwd_gemm_flags( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fwd_padding_copy( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_fwd_scratch( libxsmm_dnn_conv_config* cfg );

/**********************************************************/
/* Helper functions for BWD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_fallback_loops_bwd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_ofw_rb( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_ofh_rb( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_pixels_gemm( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_bwd_block_H( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_loop_order_bwd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_bwd_IFM( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_bwd_OFM( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_pack_input_bwd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_use_ifm_parallelization( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_rim_fmas_bwd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_blocksofm_blocking( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_init_bwd_gemm_flags( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_spread_input_bwd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_acc_load_bwd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_bwd_scratch( libxsmm_dnn_conv_config* cfg );

/**********************************************************/
/* Helper functions for UPD convolutions' parameter setup */
/**********************************************************/
LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_weight_copies_upd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_bf16_upd_algorithms( libxsmm_dnn_conv_config* inout_cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_loop_order_upd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_pack_input_upd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_avoid_rim_fmas_upd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_upd_ofw_rb( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_upd_ofh_rb( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_upd_IFM( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_block_upd_OFM( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_img_batchreduce_block( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_use_batchreduce_upd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_linearized_tasklist_upd( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_init_upd_gemm_flags( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE int libxsmm_dnn_conv_setup_upd_padding_copy( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE void libxsmm_dnn_conv_setup_upd_scratch( libxsmm_dnn_conv_config* cfg );

LIBXSMM_API_INLINE void libxsmm_dnn_conv_generate_fwd_kernels( libxsmm_dnn_conv_config* inout_cfg);

LIBXSMM_API_INLINE void libxsmm_dnn_conv_generate_bwd_kernels( libxsmm_dnn_conv_config* inout_cfg);

LIBXSMM_API_INLINE void libxsmm_dnn_conv_generate_upd_kernels( libxsmm_dnn_conv_config* inout_cfg);

LIBXSMM_API_INLINE void libxsmm_dnn_conv_free_offset_brgemm_aux_arrays( libxsmm_dnn_conv_config* cfg);

#endif

