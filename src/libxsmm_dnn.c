/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm_dnn.h>
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INTERN void libxsmm_dnn_init(int target_arch)
{
  LIBXSMM_UNUSED(target_arch);
}


LIBXSMM_API_INTERN void libxsmm_dnn_finalize(void)
{
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_get_feature_map_blocks( int C, int K, int* C_block, int* K_block, int* fm_lp_block, libxsmm_dnn_datatype datatype_in, libxsmm_dnn_datatype datatype_out ) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int ifmblock = 0;
  int ofmblock = 0;
  int lp_block = 0;
  int tmp_max_c_block = 64;
  int tmp_max_k_block = 64;
  int tmp_block = 0;

  /* init libxsmm */
  LIBXSMM_INIT

  /* C */
  if ( ((libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR) && (datatype_in == LIBXSMM_DNN_DATATYPE_BF16)) ||
       (libxsmm_target_archid < LIBXSMM_X86_AVX512 ) ) {
    tmp_max_c_block = 32;
  } else if ( libxsmm_target_archid == LIBXSMM_AARCH64_V81 ) {
    tmp_max_c_block = 16;
  }
  if ( C < tmp_max_c_block ) {
    ifmblock = C;
  } else {
    for ( tmp_block = 1; tmp_block <= tmp_max_c_block; tmp_block *= 2 ) {
      if ( C % tmp_block == 0 ) ifmblock = tmp_block;
    }
  }

  /* K */
  if ( ((libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR) && (datatype_in == LIBXSMM_DNN_DATATYPE_BF16)) ||
       (libxsmm_target_archid < LIBXSMM_X86_AVX512 ) ) {
    tmp_max_k_block = 32;
  } else if ( libxsmm_target_archid == LIBXSMM_AARCH64_V81 ) {
    tmp_max_c_block = 16;
  }
  if ( K < tmp_max_k_block ) {
    ofmblock = K;
  } else {
    for ( tmp_block = 1; tmp_block <= tmp_max_k_block; tmp_block *= 2 ) {
      if ( K % tmp_block == 0 ) ofmblock = tmp_block;
    }
  }

  /* when do we need VNNI format? */
  if ( (datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
    lp_block = 1;
  } else if ( (datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
    lp_block = 2;
  } else if ( (datatype_in == LIBXSMM_DNN_DATATYPE_I16) && ((datatype_out == LIBXSMM_DNN_DATATYPE_I32) || (datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ) {
    lp_block = 2;
  } else if (datatype_in == LIBXSMM_DNN_DATATYPE_I8) {
    lp_block = 4;
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
    return status;
  }

  *C_block = ifmblock;
  *K_block = ofmblock;
  *fm_lp_block = lp_block;

  return status;
}


LIBXSMM_API const char* libxsmm_dnn_get_error(libxsmm_dnn_err_t code)
{
  switch (code) {
    case LIBXSMM_DNN_SUCCESS:
      return "LIBXSMM DNN Success!";
    case LIBXSMM_DNN_WARN_FALLBACK:
      return "LIBXSMM DNN Warning: Falling back to naive code as target is currently not supported by LIBXSMM!";
    case LIBXSMM_DNN_WARN_RNN_SUBOPTIMAL_N_BLOCKING:
      return "LIBXSMM DNN Warning: RNN cell suboptimal minibatch blocking!";
    case LIBXSMM_DNN_WARN_RNN_SUBOPTIMAL_C_BLOCKING:
      return "LIBXSMM DNN Warning: RNN cell suboptimal input feature blocking!";
    case LIBXSMM_DNN_WARN_RNN_SUBOPTIMAL_K_BLOCKING:
      return "LIBXSMM DNN Warning: RNN cell suboptimal output feature blocking!";
    case LIBXSMM_DNN_WARN_FC_SUBOPTIMAL_N_BLOCKING:
      return "LIBXSMM DNN Warning: FC layer suboptimal minibatch blocking!";
    case LIBXSMM_DNN_WARN_FC_SUBOPTIMAL_C_BLOCKING:
      return "LIBXSMM DNN Warning: FC layer suboptimal input feature blocking!";
    case LIBXSMM_DNN_WARN_FC_SUBOPTIMAL_K_BLOCKING:
      return "LIBXSMM DNN Warning: FC layer suboptimal output feature blocking!";
    case LIBXSMM_DNN_ERR_GENERAL:
      return "LIBXSMM DNN Error: General error occurred!";
    case LIBXSMM_DNN_ERR_CREATE_HANDLE:
      return "LIBXSMM DNN Error: Handle creation failed!";
    case LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE:
      return "LIBXSMM DNN Error: Requested datatype is not available!";
    case LIBXSMM_DNN_ERR_INVALID_BLOCKING:
      return "LIBXSMM DNN Error: Requested Input/Output buffer size cannot be blocked!";
    case LIBXSMM_DNN_ERR_INVALID_HANDLE:
      return "LIBXSMM DNN Error: An invalid handle was provided!";
    case LIBXSMM_DNN_ERR_DATA_NOT_BOUND:
      return "LIBXSMM DNN Error: Not all required sources and destinations have been bound to convolution!";
    case LIBXSMM_DNN_ERR_CREATE_TENSOR:
      return "LIBXSMM DNN Error: Tensor creation failed!";
    case LIBXSMM_DNN_ERR_INVALID_TENSOR:
      return "LIBXSMM DNN Error: Invalid tensor was specified!";
    case LIBXSMM_DNN_ERR_MISMATCH_TENSOR:
      return "LIBXSMM DNN Error: Tensor doesn't match handle it should be bind to!";
    case LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR:
      return "LIBXSMM DNN Error: Invalid handle or tensor!";
    case LIBXSMM_DNN_ERR_INVALID_KIND:
      return "LIBXSMM DNN Error: Invalid convolution kind!";
    case LIBXSMM_DNN_ERR_INVALID_FORMAT_NCHW:
      return "LIBXSMM DNN Error: NCHW format is currently not natively supported by LIBXSMM!";
    case LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT:
      return "LIBXSMM DNN Error: Unsupported destination format when copying data!";
    case LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT:
      return "LIBXSMM DNN Error: Unsupported source format when copying data!";
    case LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE:
      return "LIBXSMM DNN Error: Unsupported format when requesting a convolution!";
    case LIBXSMM_DNN_ERR_INVALID_FORMAT_KCRS:
      return "LIBXSMM DNN Error: KCRS format is currently not natively supported by LIBXSMM!";
    case LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL:
      return "LIBXSMM DNN Error: Invalid format was specified!";
    case LIBXSMM_DNN_ERR_CREATE_LAYOUT:
      return "LIBXSMM DNN Error: Layout creation failed!";
    case LIBXSMM_DNN_ERR_INVALID_LAYOUT:
      return "LIBXSMM DNN Error: Invalid layout was specified!";
    case LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH:
      return "LIBXSMM DNN Error: Unsupported architecture!";
    case LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED:
      return "LIBXSMM DNN Error: scratch binding failed as scratch was not allocated!";
    case LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE:
      return "LIBXSMM DNN Error: an unknown tensor type was provided!";
    case LIBXSMM_DNN_ERR_INVALID_ALGO:
      return "LIBXSMM DNN Error: Invalid algorithm was specified!";
    case LIBXSMM_DNN_ERR_INVALID_PADDING:
      return "LIBXSMM DNN Error: Invalid padding was specified!";
    case LIBXSMM_DNN_ERR_TIME_STEPS_TOO_SMALL:
      return "LIBXSMM DNN Error: time steps should be >= 2 for RNN/LSTM!";
    case LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS:
      return "LIBXSMM DNN Error: failed to create internal layout arrays!";
    case LIBXSMM_DNN_ERR_NOT_IMPLEMENTED:
      return "LIBXSMM DNN Error: the requested functionality is right now not implemented!";
    case LIBXSMM_DNN_ERR_FUSEDBN_UNSUPPORTED_ORDER:
      return "LIBXSMM DNN Error: the requested order of fusion in batch norm is right now not implemented!";
    case LIBXSMM_DNN_ERR_FUSEDBN_UNSUPPORTED_FUSION:
      return "LIBXSMM DNN Error: the requested fusion in batch norm is right now not implemented!";
    case LIBXSMM_DNN_ERR_INVALID_FORMAT_FUSEDBN:
      return "LIBXSMM DNN Error: Unsupported format when requesting a fused batch norm!";
    case LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING:
      return "LIBXSMM DNN Error: Unsupported pooling operations was requested!";
    case LIBXSMM_DNN_ERR_INVALID_FORMAT_FC:
      return "LIBXSMM DNN Error: Unsupported format when requesting a fullyconnected layer!";
    case LIBXSMM_DNN_ERR_RNN_INVALID_SEQ_LEN:
      return "LIBXSMM DNN Error: max sequence length is shorter than sequence length we attempt to set!";
    case LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER:
      return "LIBXSMM DNN Error: the requested order of fusion in group norm is right now not implemented!";
    case LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION:
      return "LIBXSMM DNN Error: the requested fusion in group norm is right now not implemented!";
    case LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION:
      return "LIBXSMM DNN Error: the requested fusion in fullyconnected is right now not implemented!";
    default:
      return "LIBXSMM DNN Error: Unknown error or warning occurred!";
  }
}


LIBXSMM_API size_t libxsmm_dnn_typesize(libxsmm_dnn_datatype datatype)
{
  switch (datatype) {
    case LIBXSMM_DNN_DATATYPE_F32: return 4;
    case LIBXSMM_DNN_DATATYPE_I32: return 4;
    case LIBXSMM_DNN_DATATYPE_BF16: return 2;
    case LIBXSMM_DNN_DATATYPE_I16: return 2;
    case LIBXSMM_DNN_DATATYPE_I8:  return 1;
    /* no error expected as enumeration really arrives at an enum; compiler-checked */
    default: return 1;
  }
}


LIBXSMM_API size_t libxsmm_dnn_get_simd_width(libxsmm_dnn_datatype datatype)
{
  size_t l_cl_width_bytes;

  /* init libxsmm */
  LIBXSMM_INIT

  if ( libxsmm_target_archid == LIBXSMM_X86_GENERIC ||
       libxsmm_target_archid == LIBXSMM_X86_SSE3    ||
       libxsmm_target_archid == LIBXSMM_X86_SSE42 ) {
    l_cl_width_bytes = 16;
  } else if ( libxsmm_target_archid == LIBXSMM_X86_AVX2 ||
      libxsmm_target_archid == LIBXSMM_X86_AVX ) {
    l_cl_width_bytes = 32;
  } else {
    l_cl_width_bytes = 64;
  }

  return l_cl_width_bytes/libxsmm_dnn_typesize(datatype);
}

LIBXSMM_API_INLINE float libxsmm_internal_get_max( float* in_buffer, int length ) {
  float absmax_value = LIBXSMM_ABS(in_buffer[0]);
  int i = 0;
#ifdef _OPENMP
  LIBXSMM_OMP_VAR(i);
# pragma omp parallel private(i)
  {
    float my_absmax_value = absmax_value;
#   pragma omp for
    for (i = 0; i < length; ++i ) {
      if (LIBXSMM_ABS(in_buffer[i]) > my_absmax_value) {
        my_absmax_value = LIBXSMM_ABS(in_buffer[i]);
      }
    }
#   pragma omp critical
    {
      if (my_absmax_value > absmax_value) {
        absmax_value = my_absmax_value;
      }
    }
  }
#else
  for (i = 1; i < length; ++i ) {
    if (LIBXSMM_ABS(in_buffer[i]) > absmax_value) {
      absmax_value = LIBXSMM_ABS(in_buffer[i]);
    }
  }
#endif

  return absmax_value;
}


LIBXSMM_API_INLINE unsigned char libxsmm_internal_get_max_exp( float* in_buffer, int length ) {
  libxsmm_intfloat val_exp;
  unsigned char max_exp = 0;

  /* bit-wise conversion to int */
  val_exp.f = libxsmm_internal_get_max( in_buffer, length );
  /* shift by mantissa to the right and convert to char */
  max_exp = (unsigned char)((val_exp.ui & LIBXSMM_DNN_MASK_ABS_F32) >> LIBXSMM_DNN_MANT_SZ_F32);

  return max_exp;
}


LIBXSMM_API_INLINE short libxsmm_internal_quantize_scalar_no_scf( float input, unsigned char max_exp, unsigned char add_shift, int round_mode ) {
  libxsmm_intfloat value;
  unsigned int qvalue = 0;
  unsigned int mant = 0;
  unsigned int sign = 0;
  unsigned char rhs = 0;
  unsigned char exp_off = 0;

  /* init libxsmm */
  LIBXSMM_INIT

  /* in case of zero we don't need to do anything */
  if (LIBXSMM_FEQ(input, 0)) {
    qvalue = 0;
  } else {
    /* let's get a float copy to work on */
    /* vinp = LIBXSMM_INTRINSICS_MM512_LOAD_PS( in_buffer[i] ); */
    value.f = input;
    /* let's compute the offset of the current exp at pos i from max offset, we need to mask the sign bit though */
    /*__m512i vexp     = _mm512_cvtps_epi32(_mm512_getexp_ps (vinp));
      __m512i vexp_off = _mm512_sub_epi32(maxexpf, vexp);*/
    exp_off = (unsigned char)(max_exp - ((value.ui & LIBXSMM_DNN_MASK_ABS_F32) >> LIBXSMM_DNN_MANT_SZ_F32));
    /* cut out mantissa and set leading bit */
    /*__m512i mmask = _mm512_set1_epi32(LIBXSMM_DNN_MASK_MANT_F32);
      __m512i vmant = _mm512_or_epi32(_mm512_set1_epi32(0x1 << LIBXSMM_DNN_MANT_SZ_F32), _mm512_and_epi32( _mm512_castps_si512( vinp ), mmask));*/
    mant = ((0x1 << LIBXSMM_DNN_MANT_SZ_F32) | (value.ui & LIBXSMM_DNN_MASK_MANT_F32));
    /* extract sign */
    /* __mmask16 smask =  _mm512_cmplt_ps_mask (inp, _mm512_set1_ps(0)); */
    sign = ((value.ui & LIBXSNN_DNN_MASK_SIGN_F32) >> (LIBXSMM_DNN_SZ_F32-1));
    /* calculate rhs, be aware of the now explicit leading bit, @TODO add DFP8/4 */
    rhs = (unsigned char)((LIBXSMM_DNN_MANT_SZ_F32+1) - LIBXSMM_DNN_MANT_DFP16 + exp_off + add_shift);
    /* some safety, to generate 0 when we fall off quant region, @TODO issue a LIBXSMM WARNING: that we shifted out the entire mantissa */
    if (rhs > (LIBXSMM_DNN_MANT_SZ_F32+1)) {
      rhs = (LIBXSMM_DNN_MANT_SZ_F32+1);
    }
    /* finally shift the value into the region we need, this is now a 15-add_rhs bit number for the max value in in_buffer */
    qvalue = (mant >> rhs);
    /* handle sign, 2 complement */
    if ( (sign > 0) && (qvalue > 0) ) {
      qvalue = (~qvalue + 1);
    }

    if (round_mode == LIBXSMM_DNN_QUANT_BIAS_ROUND) {
      /* biased rounding towards next bigger number */
      /* first let's determine in the original number if we need a bias rounding, @TODO need fix for F64 */
      int bias_needed = (mant & (0x3 << (rhs-2)));
      /* apply bias */
      if (bias_needed > 0) {
        qvalue++;
      }
    } else if (round_mode == LIBXSMM_DNN_QUANT_NEAREST_ROUND) {
      int nearest_needed = (mant & (0x1 << (rhs-1)));
      /* apply rounding */
      if ((nearest_needed > 0) && (rhs > 1)) {
        qvalue++;
      }
    } else if (round_mode == LIBXSMM_DNN_QUANT_STOCH_ROUND) {
      /* stochastic rounding, as implemented in the IBM paper from 2015, @TODO, fix F64 and DFP8 */
      const float eps = LIXSMMM_DNN_RES_DFP16;
      /* coverity[dont_call] */
      const float r = (float)rand();
      libxsmm_intfloat fvalue;
      float p, q;
      /* masking all bits which will be shifted out */
      fvalue.ui = value.ui & ((LIBXSMM_DNN_MASK_FULL_F32) << rhs);
      /* drawing a random number */
      p = r/((float)RAND_MAX);
      q = (input - fvalue.f)/eps;
      /* apply rounding if needed */
      if ((p + q) > 0.5f) {
        ++qvalue;
      }
    } else {
      /* do nothing about rounding, just chop */
    }
  }

  return (short)qvalue;
}


/* @TODO make this routine aware of any int type */
LIBXSMM_API void libxsmm_dnn_quantize( float* in_buffer, short* out_buffer, int length, unsigned char add_shift, unsigned char* scf, int round_mode ) {
  int i = 0;

  /* init libxsmm */
  LIBXSMM_INIT

  /* in case we are using FP-Mul based quantization we use a different path for now
     @TODO let's unify the paths by using the similar vectorization for both */
  if ( round_mode == LIBXSMM_DNN_QUANT_FPHW_ROUND ) {
    const float max_value = libxsmm_internal_get_max( in_buffer, length );
    int maxexp = 0;
    /* take return value of LIBXSMM_FREXPF to mute static analysis issue */
    float scfq = LIBXSMM_FREXPF(max_value, &maxexp);
    maxexp -= (15/*LIBXSMM_DNN_MANT_DFP16?*/ - add_shift);
    scfq = libxsmm_sexp2_i8i(-maxexp);

#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    if ( length % 16 == 0 ) {
      __m512 vscfq = _mm512_set1_ps(scfq);
#ifdef _OPENMP
#     pragma omp parallel for private(i)
#endif
      for (i = 0; i < length; i+=16 ) {
        _mm256_stream_si256( (__m256i *)&(out_buffer[i]), LIBXSMM_INTRINSICS_MM512_QUANTIZE_NEAR_PS_EPI16( &(in_buffer[i]), vscfq ) );
      }
    } else {
#endif
#ifdef _OPENMP
#     pragma omp parallel for private(i)
#endif
      for (i = 0; i < length; ++i ) {
        out_buffer[i] = (short)LIBXSMM_ROUNDF(in_buffer[i] * scfq);
      }
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    }
#endif
    /* @TODO, we need to potentially fix this unsigned char problem */
#if !defined(NDEBUG) /* library code is expected to be mute */
    if (maxexp > 0) {
      fprintf(stderr, "error quant fil\n");
    }
#endif
    *scf = (unsigned char)(-maxexp);
  } else {
    /* get max exponent */
    unsigned char max_exp = libxsmm_internal_get_max_exp( in_buffer, length );

    /* if we go for stochastic rounding, let's initialize random seed */
    if ( round_mode == LIBXSMM_DNN_QUANT_STOCH_ROUND ) {
      srand(libxsmm_timer_tick() % ((unsigned int)-1));
    }

#ifdef _OPENMP
#   pragma omp parallel for private(i)
#endif
    for (i = 0; i < length; ++i ) {
      out_buffer[i] = libxsmm_internal_quantize_scalar_no_scf( in_buffer[i], max_exp, add_shift, round_mode );
    }

    *scf = (unsigned char)(14 - add_shift - (max_exp - 127));
  }
}


LIBXSMM_API void libxsmm_dnn_quantize_act( float* in_buffer, short* out_buffer, unsigned int N, unsigned int C, unsigned int H, unsigned int W, unsigned int cblk_f32, unsigned int cblk_i16, unsigned int lp_blk, unsigned char add_shift, unsigned char* scf, int round_mode ) {
  LIBXSMM_VLA_DECL(5, const float, in,  in_buffer,  C/cblk_f32, H, W, cblk_f32);
  LIBXSMM_VLA_DECL(6, short, out, out_buffer, C/(cblk_i16*lp_blk), H, W, cblk_i16, lp_blk);
  const unsigned int cblk = C/(cblk_i16*lp_blk);
  int i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5, i6;

  /* init libxsmm */
  LIBXSMM_INIT

  /* some quick and dirty checks */
  assert((C % cblk_f32) == 0);
  assert((C % cblk_i16) == 0);

  /* in case we are using FP-Mul based quantization we use a different path for now
     @TODO let's unify the paths by using the similar vectorization for both */
  if ( round_mode == LIBXSMM_DNN_QUANT_FPHW_ROUND ) {
    const float max_value = libxsmm_internal_get_max( in_buffer, N*C*H*W );
    int maxexp = 0;
    /* take return value of LIBXSMM_FREXPF to mute static analysis issue */
    float scfq = LIBXSMM_FREXPF(max_value, &maxexp);
    maxexp -= (15/*LIBXSMM_DNN_MANT_DFP16?*/ - add_shift);
    scfq = libxsmm_sexp2_i8i(-maxexp);

#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    if ( (cblk_f32 == 16) && (cblk_i16*lp_blk == 16) ) {
      __m512 vscfq = _mm512_set1_ps(scfq);
#ifdef _OPENMP
      LIBXSMM_OMP_VAR(i1);
#     pragma omp parallel for private(i1)
#endif
      for (i1 = 0; i1 < (int)(N*C*H*W); i1 += 16 ) {
        _mm256_stream_si256( (__m256i *)&(out_buffer[i1]), LIBXSMM_INTRINSICS_MM512_QUANTIZE_NEAR_PS_EPI16( &(in_buffer[i1]), vscfq ) );
      }
    } else {
#endif
#ifdef _OPENMP
      LIBXSMM_OMP_VAR(i1); LIBXSMM_OMP_VAR(i2); LIBXSMM_OMP_VAR(i3); LIBXSMM_OMP_VAR(i4); LIBXSMM_OMP_VAR(i5); LIBXSMM_OMP_VAR(i6);
#     pragma omp parallel for private(i1, i2, i3, i4, i5, i6) LIBXSMM_OPENMP_COLLAPSE(4)
#endif
      for (i1 = 0; i1 < (int)N; ++i1 ) {
        for (i2 = 0; i2 < (int)cblk; ++i2 ) {
          for (i3 = 0; i3 < (int)H; ++i3 ) {
            for (i4 = 0; i4 < (int)W; ++i4 ) {
              for (i5 = 0; i5 < (int)cblk_i16; ++i5 ) {
                for (i6 = 0; i6 < (int)lp_blk; ++i6 ) {
                  const int fi1 = i1;
                  const int fi2 = ((i2*cblk_i16*lp_blk)+(i5*lp_blk)+i6)/cblk_f32;
                  const int fi3 = i3;
                  const int fi4 = i4;
                  const int fi5 = ((i2*cblk_i16*lp_blk)+(i5*lp_blk)+i6)%cblk_f32;
                  LIBXSMM_VLA_ACCESS(6, out, i1, i2, i3, i4, i5, i6, cblk, H, W, cblk_i16, lp_blk) = (short)LIBXSMM_ROUNDF(
                  LIBXSMM_VLA_ACCESS(5, in, fi1, fi2, fi3, fi4, fi5, C / cblk_f32, H, W, cblk_f32) * scfq);
                }
              }
            }
          }
        }
      }
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    }
#endif
    /* @TODO, we need to potentially fix this unsigned char problem */
#if !defined(NDEBUG) /* library code is expected to be mute */
    if (maxexp > 0) {
      fprintf(stderr, "error quant act\n");
    }
#endif
    *scf = (unsigned char)(-maxexp);
  } else {
    /* get max exponent */
    unsigned char max_exp = libxsmm_internal_get_max_exp( in_buffer, N*C*H*W );

    /* if we go for stochastic rounding, let's initialize random seed */
    if ( round_mode == LIBXSMM_DNN_QUANT_STOCH_ROUND ) {
      srand(libxsmm_timer_tick() % ((unsigned int)-1));
    }

#ifdef _OPENMP
#   pragma omp parallel for private(i1, i2, i3, i4, i5, i6) LIBXSMM_OPENMP_COLLAPSE(4)
#endif
    for (i1 = 0; i1 < (int)N; ++i1 ) {
      for (i2 = 0; i2 < (int)cblk; ++i2 ) {
        for (i3 = 0; i3 < (int)H; ++i3 ) {
          for (i4 = 0; i4 < (int)W; ++i4 ) {
            for (i5 = 0; i5 < (int)cblk_i16; ++i5 ) {
              for (i6 = 0; i6 < (int)lp_blk; ++i6 ) {
                const int fi1 = i1;
                const int fi2 = ((i2*cblk_i16*lp_blk)+(i5*lp_blk)+i6)/cblk_f32;
                const int fi3 = i3;
                const int fi4 = i4;
                const int fi5 = ((i2*cblk_i16*lp_blk)+(i5*lp_blk)+i6)%cblk_f32;
                LIBXSMM_VLA_ACCESS(6, out, i1, i2, i3, i4, i5, i6, cblk, H, W, cblk_i16, lp_blk) = libxsmm_internal_quantize_scalar_no_scf(
                LIBXSMM_VLA_ACCESS(5, in, fi1, fi2, fi3, fi4, fi5, C / cblk_f32, H, W, cblk_f32), max_exp, add_shift, round_mode);
              }
            }
          }
        }
      }
    }

    *scf = (unsigned char)(14 - add_shift - (max_exp - 127));
  }
}


LIBXSMM_API void libxsmm_dnn_quantize_fil( float* in_buffer, short* out_buffer, unsigned int K, unsigned int C, unsigned int R, unsigned int S, unsigned int cblk_f32, unsigned int cblk_i16, unsigned int kblk_f32, unsigned int kblk_i16, unsigned int lp_blk, unsigned char add_shift, unsigned char* scf, int round_mode ) {
  LIBXSMM_VLA_DECL(6, const float, in,  in_buffer,  C/cblk_f32, R, S, cblk_f32, kblk_f32);
  LIBXSMM_VLA_DECL(7, short, out, out_buffer, C/(cblk_i16*lp_blk), R, S, cblk_i16, kblk_i16, lp_blk);
  unsigned int cblk = C/(cblk_i16*lp_blk);
  unsigned int kblk = K/kblk_i16;
  int i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5, i6, i7;

  /* some quick and dirty checks */
  assert((C % cblk_f32) == 0);
  assert((C % (cblk_i16*lp_blk)) == 0);
  assert((K % kblk_f32) == 0);
  assert((K % kblk_i16) == 0);
  assert((lp_blk % 2) == 0);

  /* init libxsmm */
  LIBXSMM_INIT

  /* in case we are using FP-Mul based quantization we use a different path for now
     @TODO let's unify the paths by using the similar vectorization for both */
  if ( round_mode == LIBXSMM_DNN_QUANT_FPHW_ROUND ) {
    const float max_value = libxsmm_internal_get_max( in_buffer, K*C*R*S );
    int maxexp = 0;
    /* take return value of LIBXSMM_FREXPF to mute static analysis issue */
    float scfq = LIBXSMM_FREXPF(max_value, &maxexp);
    maxexp -= (15/*LIBXSMM_DNN_MANT_DFP16?*/ - add_shift);
    scfq = libxsmm_sexp2_i8i(-maxexp);

#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    if ( (kblk_f32 == 16) && (cblk_f32 == 16) && (kblk_i16 == 16) && (cblk_i16*lp_blk == 16) ) {
      const __m512 vscfq = _mm512_set1_ps(scfq);
      const __m512i permute_compact_idx = _mm512_set_epi32(15,14,13,12,7,6,5,4,11,10,9,8,3,2,1,0);
#ifdef _OPENMP
#     pragma omp parallel for private(i1, i2, i3, i4, i5) LIBXSMM_OPENMP_COLLAPSE(4)
#endif
      for (i1 = 0; i1 < (int)kblk; ++i1 ) {
        for (i2 = 0; i2 < (int)cblk; ++i2 ) {
          for (i3 = 0; i3 < (int)R; ++i3 ) {
            for (i4 = 0; i4 < (int)S; ++i4 ) {
              for (i5 = 0; i5 < 16; i5+=2 ) {
                __m256i even_ch = LIBXSMM_INTRINSICS_MM512_QUANTIZE_NEAR_PS_EPI16(
                  &LIBXSMM_VLA_ACCESS(6, in, i1, i2, i3, i4, i5 + 0, 0, C / cblk_f32, R, S, cblk_f32, kblk_f32), vscfq);
                __m256i odd_ch  = LIBXSMM_INTRINSICS_MM512_QUANTIZE_NEAR_PS_EPI16(
                  &LIBXSMM_VLA_ACCESS(6, in, i1, i2, i3, i4, i5 + 1, 0, C / cblk_f32, R, S, cblk_f32, kblk_f32), vscfq);
                __m256i compressed_lo = _mm256_unpacklo_epi16(even_ch, odd_ch);
                __m256i compressed_hi = _mm256_unpackhi_epi16(even_ch, odd_ch);
                __m512i compact =  _mm512_inserti64x4( _mm512_setzero_si512(), compressed_lo, 0);
                compact = _mm512_inserti64x4(compact, compressed_hi, 1);
                compact = _mm512_permutexvar_epi32(permute_compact_idx, compact);
                LIBXSMM_INTRINSICS_MM512_STREAM_SI512(
                  (void*)&LIBXSMM_VLA_ACCESS(7, out, i1, i2, i3, i4, i5 / 2, 0, 0, cblk, R, S, cblk_i16, kblk_i16, lp_blk),
                  compact);
              }
            }
          }
        }
      }
    } else {
#endif
#ifdef _OPENMP
      LIBXSMM_OMP_VAR(i1); LIBXSMM_OMP_VAR(i2); LIBXSMM_OMP_VAR(i3); LIBXSMM_OMP_VAR(i4); LIBXSMM_OMP_VAR(i5); LIBXSMM_OMP_VAR(i6); LIBXSMM_OMP_VAR(i7);
#     pragma omp parallel for private(i1, i2, i3, i4, i5, i6, i7) LIBXSMM_OPENMP_COLLAPSE(4)
#endif
      for (i1 = 0; i1 < (int)kblk; ++i1 ) {
        for (i2 = 0; i2 < (int)cblk; ++i2 ) {
          for (i3 = 0; i3 < (int)R; ++i3 ) {
            for (i4 = 0; i4 < (int)S; ++i4 ) {
              for (i5 = 0; i5 < (int)cblk_i16; ++i5 ) {
                for (i6 = 0; i6 < (int)kblk_i16; ++i6 ) {
                  for (i7 = 0; i7 < (int)lp_blk; ++i7 ) {
                    const int fi1 = ((i1*kblk_i16)+i6)/kblk_f32;
                    const int fi2 = ((i2*cblk_i16*lp_blk)+(i5*lp_blk)+i7)/cblk_f32;
                    const int fi3 = i3;
                    const int fi4 = i4;
                    const int fi5 = ((i2*cblk_i16*lp_blk)+(i5*lp_blk)+i7)%cblk_f32;
                    const int fi6 = ((i1*kblk_i16)+i6)%kblk_f32;
                    LIBXSMM_VLA_ACCESS(7, out, i1, i2, i3, i4, i5, i6, i7, cblk, R, S, cblk_i16, kblk_i16, lp_blk) = (short)LIBXSMM_ROUNDF(
                    LIBXSMM_VLA_ACCESS(6, in, fi1, fi2, fi3, fi4, fi5, fi6, C / cblk_f32, R, S, cblk_f32, kblk_f32) * scfq);
                  }
                }
              }
            }
          }
        }
      }
#if (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
    }
#endif
    /* @TODO, we need to potentially fix this unsigned char problem */
#if !defined(NDEBUG) /* library code is expected to be mute */
    if (maxexp > 0) {
      fprintf(stderr, "error quant fil\n");
    }
#endif
    *scf = (unsigned char)(-maxexp);
  } else {
    /* get max exponent */
    unsigned char max_exp = libxsmm_internal_get_max_exp( in_buffer, K*C*R*S );

    /* if we go for stochastic rounding, let's initialize random seed */
    if ( round_mode == LIBXSMM_DNN_QUANT_STOCH_ROUND ) {
      srand(libxsmm_timer_tick() % ((unsigned int)-1));
    }

#ifdef _OPENMP
#   pragma omp parallel for private(i1, i2, i3, i4, i5, i6, i7) LIBXSMM_OPENMP_COLLAPSE(4)
#endif
    for (i1 = 0; i1 < (int)kblk; ++i1 ) {
      for (i2 = 0; i2 < (int)cblk; ++i2 ) {
        for (i3 = 0; i3 < (int)R; ++i3 ) {
          for (i4 = 0; i4 < (int)S; ++i4 ) {
            for (i5 = 0; i5 < (int)cblk_i16; ++i5 ) {
              for (i6 = 0; i6 < (int)kblk_i16; ++i6 ) {
                for (i7 = 0; i7 < (int)lp_blk; ++i7 ) {
                  const int fi1 = ((i1*kblk_i16)+i6)/kblk_f32;
                  const int fi2 = ((i2*cblk_i16*lp_blk)+(i5*lp_blk)+i7)/cblk_f32;
                  const int fi3 = i3;
                  const int fi4 = i4;
                  const int fi5 = ((i2*cblk_i16*lp_blk)+(i5*lp_blk)+i7)%cblk_f32;
                  const int fi6 = ((i1*kblk_i16)+i6)%kblk_f32;
                  LIBXSMM_VLA_ACCESS(7, out, i1, i2, i3, i4, i5, i6, i7, cblk, R, S, cblk_i16, kblk_i16, lp_blk) = libxsmm_internal_quantize_scalar_no_scf(
                  LIBXSMM_VLA_ACCESS(6, in, fi1, fi2, fi3, fi4, fi5, fi6, C / cblk_f32, R, S, cblk_f32, kblk_f32), max_exp, add_shift, round_mode);
                }
              }
            }
          }
        }
      }
    }

    *scf = (unsigned char)(14 - add_shift - (max_exp - 127));
  }
}


LIBXSMM_API void libxsmm_dnn_dequantize( short* in_buffer, float* out_buffer, int length, unsigned char scf ) {
  const float val_exp = libxsmm_sexp2_i8i(-scf);
  int i = 0;

#ifdef _OPENMP
# pragma omp parallel for private(i)
#endif
  for ( i = 0; i < length; ++i ) {
    out_buffer[i] = ((float)in_buffer[i])*val_exp;
  }
}


LIBXSMM_API void libxsmm_truncate_convert_f32_bf16(const float* in, libxsmm_bfloat16* out, unsigned int length) {
  unsigned int i = 0;

  /* truncate buffer to bf16 */
  for ( i = 0; i < length; ++i ) {
    libxsmm_bfloat16_hp t;

    t.f = in[i];
    out[i] = t.i[1];
  }
}


LIBXSMM_API void libxsmm_rnaz_convert_fp32_bf16(const float* in, libxsmm_bfloat16* out, unsigned int len) {
  unsigned int i = 0;

  /* truncate buffer to bf16 */
  for ( i = 0; i < len; ++i ) {
    unsigned int int_round = 0;
    unsigned int do_round = 1;

    int_round = *((unsigned int*)&(in[i]));

    /* we don't round NaN and inf */
    if ( (int_round & 0x7f800000) == 0x7f800000 ) {
      do_round = 0;
    }

    /* perform round nearest tie away from zero */
    if ( do_round != 0 ) {
      int_round = int_round + 0x00008000;
    }

    /* create the bf16 value by shifting out the lower 16bits */
    int_round = int_round >> 16;

    out[i] = (libxsmm_bfloat16)int_round;
  }
}


LIBXSMM_API void libxsmm_rne_convert_fp32_bf16(const float* in, libxsmm_bfloat16* out, unsigned int len) {
  unsigned int i = 0;

  /* truncate buffer to bf16 */
  for ( i = 0; i < len; ++i ) {
    unsigned int int_round = 0;
    unsigned int do_round = 1;

    int_round = *((unsigned int*)&(in[i]));

    /* we don't round NaN and inf */
    if ( (int_round & 0x7f800000) == 0x7f800000 ) {
      do_round = 0;
    }

    /* perform round nearest tie even */
    if ( do_round != 0 ) {
      unsigned int fixup = (int_round >> 16) & 1;
      int_round = int_round + 0x00007fff + fixup;
    }

    /* create the bf16 value by shifting out the lower 16bits */
    int_round = int_round >> 16;

    out[i] = (unsigned short)int_round;
  }
}


LIBXSMM_API void libxsmm_convert_bf16_f32(const libxsmm_bfloat16* in, float* out, unsigned int length) {
  unsigned int i = 0;

  /* up-convert is super simple */
  for ( i = 0; i < length; ++i ) {
    libxsmm_bfloat16_hp t;

    t.i[1] = in[i];
    t.i[0] = 0;
    out[i] = t.f;
  }
}

