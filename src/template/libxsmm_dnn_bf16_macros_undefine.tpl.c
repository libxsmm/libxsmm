/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#undef LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16
#undef LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32
#undef LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH
#undef LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH
#undef _mm512_loadcvt_bf16_fp32
#undef _mm512_storecvt_fp32_bf16
#undef _mm512_streamstorecvt_fp32_bf16

#ifdef USE_CLDEMOTE
#undef USE_CLDEMOTE
#endif

#ifdef WR_PREFETCH_OUTPUT
#undef prefetchwt_chunk
#undef WR_PREFETCH_OUTPUT
#endif

