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
#include "../src/generator_aarch64_instructions.c"
#include "../src/generator_common.c"
#include "../src/generator_common_aarch64.c"
#include "../src/generator_common_x86.c"
#include "../src/generator_gemm.c"
#include "../src/generator_gemm_aarch64.c"
#include "../src/generator_gemm_amx.c"
#include "../src/generator_gemm_amx_microkernel.c"
#include "../src/generator_gemm_avx2_microkernel.c"
#include "../src/generator_gemm_avx512_microkernel.c"
#include "../src/generator_gemm_avx_microkernel.c"
#include "../src/generator_gemm_common.c"
#include "../src/generator_gemm_common_aarch64.c"
#include "../src/generator_gemm_noarch.c"
#include "../src/generator_gemm_sse_avx_avx2_avx512.c"
#include "../src/generator_gemm_sse_microkernel.c"
#include "../src/generator_mateltwise.c"
#include "../src/generator_mateltwise_aarch64.c"
#include "../src/generator_mateltwise_common.c"
#include "../src/generator_mateltwise_gather_scatter_aarch64.c"
#include "../src/generator_mateltwise_gather_scatter_avx_avx512.c"
#include "../src/generator_mateltwise_misc_aarch64.c"
#include "../src/generator_mateltwise_misc_avx_avx512.c"
#include "../src/generator_mateltwise_reduce_aarch64.c"
#include "../src/generator_mateltwise_reduce_avx_avx512.c"
#include "../src/generator_mateltwise_sse_avx_avx512.c"
#include "../src/generator_mateltwise_transform_aarch64_asimd.c"
#include "../src/generator_mateltwise_transform_aarch64_sve.c"
#include "../src/generator_mateltwise_transform_avx.c"
#include "../src/generator_mateltwise_transform_avx512.c"
#include "../src/generator_mateltwise_transform_common.c"
#include "../src/generator_mateltwise_transform_common_x86.c"
#include "../src/generator_mateltwise_transform_sse.c"
#include "../src/generator_mateltwise_unary_binary_aarch64.c"
#include "../src/generator_mateltwise_unary_binary_avx_avx512.c"
#include "../src/generator_matequation.c"
#include "../src/generator_matequation_aarch64.c"
#include "../src/generator_matequation_avx_avx512.c"
#include "../src/generator_matequation_regblocks_aarch64.c"
#include "../src/generator_matequation_regblocks_avx_avx512.c"
#include "../src/generator_matequation_scratch_aarch64.c"
#include "../src/generator_matequation_scratch_avx_avx512.c"
#include "../src/generator_packed_gemm.c"
#include "../src/generator_packed_gemm_aarch64.c"
#include "../src/generator_packed_gemm_ac_rm.c"
#include "../src/generator_packed_gemm_ac_rm_aarch64.c"
#include "../src/generator_packed_gemm_ac_rm_avx_avx2_avx512.c"
#include "../src/generator_packed_gemm_avx_avx2_avx512.c"
#include "../src/generator_packed_gemm_bc_rm.c"
#include "../src/generator_packed_gemm_bc_rm_aarch64.c"
#include "../src/generator_packed_gemm_bc_rm_avx_avx2_avx512.c"
#include "../src/generator_packed_spgemm.c"
#include "../src/generator_packed_spgemm_bcsc_bsparse.c"
#include "../src/generator_packed_spgemm_bcsc_bsparse_aarch64.c"
#include "../src/generator_packed_spgemm_bcsc_bsparse_avx_avx2_avx512_amx.c"
#include "../src/generator_packed_spgemm_csc_bsparse.c"
#include "../src/generator_packed_spgemm_csc_bsparse_aarch64.c"
#include "../src/generator_packed_spgemm_csc_bsparse_avx_avx2_avx512.c"
#include "../src/generator_packed_spgemm_csc_csparse.c"
#include "../src/generator_packed_spgemm_csc_csparse_avx_avx2_avx512.c"
#include "../src/generator_packed_spgemm_csr_asparse.c"
#include "../src/generator_packed_spgemm_csr_asparse_aarch64.c"
#include "../src/generator_packed_spgemm_csr_asparse_avx_avx2_avx512.c"
#include "../src/generator_packed_spgemm_csr_bsparse.c"
#include "../src/generator_packed_spgemm_csr_bsparse_aarch64.c"
#include "../src/generator_packed_spgemm_csr_bsparse_avx_avx2_avx512.c"
#include "../src/generator_spgemm.c"
#include "../src/generator_spgemm_csc_asparse.c"
#include "../src/generator_spgemm_csc_bsparse.c"
#include "../src/generator_spgemm_csc_reader.c"
#include "../src/generator_spgemm_csr_asparse.c"
#include "../src/generator_spgemm_csr_asparse_reg.c"
#include "../src/generator_spgemm_csr_reader.c"
#include "../src/generator_x86_instructions.c"
#include "../src/libxsmm_barrier.c"
#include "../src/libxsmm_cpuid_arm.c"
#include "../src/libxsmm_cpuid_x86.c"
#include "../src/libxsmm_ext.c"
#include "../src/libxsmm_ext_gemm.c"
#include "../src/libxsmm_ext_xcopy.c"
#include "../src/libxsmm_fsspmdm.c"
#include "../src/libxsmm_gemm.c"
#include "../src/libxsmm_generator.c"
#include "../src/libxsmm_hash.c"
#include "../src/libxsmm_lpflt_quant.c"
#include "../src/libxsmm_main.c"
#include "../src/libxsmm_malloc.c"
#include "../src/libxsmm_math.c"
#include "../src/libxsmm_matrixeqn.c"
#include "../src/libxsmm_memory.c"
#include "../src/libxsmm_mhd.c"
#include "../src/libxsmm_perf.c"
#include "../src/libxsmm_rng.c"
#include "../src/libxsmm_sync.c"
#include "../src/libxsmm_timer.c"
#include "../src/libxsmm_trace.c"
#include "../src/libxsmm_utils.c"
#include "../src/libxsmm_xcopy.c"

#endif /*LIBXSMM_SOURCE_H*/
