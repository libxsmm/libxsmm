/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_SOURCE_H
#define LIBXSMM_SOURCE_H

#if defined(LIBXSMM_MACROS_H)
# error Please do not include any LIBXSMM header other than libxsmm_source.h!
#endif
#if defined(LIBXSMM_BUILD)
# error LIBXSMM_BUILD cannot be defined for the header-only LIBXSMM!
#endif

/**
 * This header is intentionally called "libxsmm_source.h" since the followings block
 * includes *internal* files, and thereby exposes LIBXSMM's implementation.
 * The so-called "header-only" usage model gives up the clearly defined binary interface
 * (including support for hot-fixes after deployment), and requires to rebuild client
 * code for every (internal) change of LIBXSMM. Please make sure to only rely on the
 * public interface as the internal implementation may change without notice.
 */
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include "../src/generator_common.c"
#include "../src/generator_gemm.c"
#include "../src/generator_gemm_avx2_microkernel.c"
#include "../src/generator_gemm_avx512_microkernel.c"
#include "../src/generator_gemm_avx_microkernel.c"
#include "../src/generator_gemm_common.c"
#include "../src/generator_gemm_noarch.c"
#include "../src/generator_gemm_sse3_avx_avx2_avx512.c"
#include "../src/generator_gemm_sse3_microkernel.c"
#include "../src/generator_matcopy.c"
#include "../src/generator_matcopy_avx_avx512.c"
#include "../src/generator_packed.c"
#include "../src/generator_packed_gemm_ac_rm_avx_avx2_avx512.c"
#include "../src/generator_packed_gemm_avx_avx512.c"
#include "../src/generator_packed_gemm_bc_rm_avx_avx2_avx512.c"
#include "../src/generator_packed_getrf_avx_avx512.c"
#include "../src/generator_packed_trmm_avx_avx512.c"
#include "../src/generator_packed_trsm_avx_avx512.c"
#include "../src/generator_spgemm.c"
#include "../src/generator_spgemm_csc_asparse.c"
#include "../src/generator_spgemm_csc_bsparse.c"
#include "../src/generator_spgemm_csc_bsparse_soa.c"
#include "../src/generator_spgemm_csc_csparse_soa.c"
#include "../src/generator_spgemm_csc_reader.c"
#include "../src/generator_spgemm_csr_asparse.c"
#include "../src/generator_spgemm_csr_asparse_reg.c"
#include "../src/generator_spgemm_csr_asparse_soa.c"
#include "../src/generator_spgemm_csr_bsparse_soa.c"
#include "../src/generator_spgemm_csr_reader.c"
#include "../src/generator_transpose.c"
#include "../src/generator_transpose_avx_avx512.c"
#include "../src/generator_x86_instructions.c"
#include "../src/libxsmm_blocked_gemm.c"
#include "../src/libxsmm_cpuid_x86.c"
#include "../src/libxsmm_dnn.c"
#include "../src/libxsmm_dnn_convolution.c"
#include "../src/libxsmm_dnn_convolution_backward.c"
#include "../src/libxsmm_dnn_convolution_forward.c"
#include "../src/libxsmm_dnn_convolution_weight_update.c"
#include "../src/libxsmm_dnn_elementwise.c"
#include "../src/libxsmm_dnn_fullyconnected.c"
#include "../src/libxsmm_dnn_fullyconnected_backward_weight_update.c"
#include "../src/libxsmm_dnn_fullyconnected_forward.c"
#include "../src/libxsmm_dnn_fusedbatchnorm.c"
#include "../src/libxsmm_dnn_fusedbatchnorm_backward.c"
#include "../src/libxsmm_dnn_fusedbatchnorm_forward.c"
#include "../src/libxsmm_dnn_fusedgroupnorm.c"
#include "../src/libxsmm_dnn_fusedgroupnorm_backward.c"
#include "../src/libxsmm_dnn_fusedgroupnorm_forward.c"
#include "../src/libxsmm_dnn_optimizer.c"
#include "../src/libxsmm_dnn_optimizer_sgd.c"
#include "../src/libxsmm_dnn_pooling.c"
#include "../src/libxsmm_dnn_pooling_backward.c"
#include "../src/libxsmm_dnn_pooling_forward.c"
#include "../src/libxsmm_dnn_rnncell.c"
#include "../src/libxsmm_dnn_rnncell_backward_weight_update.c"
#include "../src/libxsmm_dnn_rnncell_forward.c"
#include "../src/libxsmm_dnn_softmaxloss.c"
#include "../src/libxsmm_dnn_softmaxloss_backward.c"
#include "../src/libxsmm_dnn_softmaxloss_forward.c"
#include "../src/libxsmm_dnn_tensor.c"
#include "../src/libxsmm_ext.c"
#include "../src/libxsmm_ext_blocked_gemm.c"
#include "../src/libxsmm_ext_gemm.c"
#include "../src/libxsmm_ext_xcopy.c"
#include "../src/libxsmm_fsspmdm.c"
#include "../src/libxsmm_gemm.c"
#include "../src/libxsmm_generator.c"
#include "../src/libxsmm_hash.c"
#include "../src/libxsmm_main.c"
#include "../src/libxsmm_malloc.c"
#include "../src/libxsmm_math.c"
#include "../src/libxsmm_memory.c"
#include "../src/libxsmm_mhd.c"
#include "../src/libxsmm_perf.c"
#include "../src/libxsmm_python.c"
#include "../src/libxsmm_rng.c"
#include "../src/libxsmm_spmdm.c"
#include "../src/libxsmm_sync.c"
#include "../src/libxsmm_timer.c"
#include "../src/libxsmm_trace.c"
#include "../src/libxsmm_xcopy.c"
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#endif /*LIBXSMM_SOURCE_H*/
