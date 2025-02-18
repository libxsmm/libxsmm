/******************************************************************************
* Copyright (c), 2025 IBM Corporation - All rights reserved.                  *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Will Trojak (IBM Corp.)
******************************************************************************/

#include "generator_s390x_instructions.h"

/* Based on "z/Architecture: Principles of Operation".
   Below is a table showing Z model corresponding revision number:
   arch11 (z14)     SA22-7832-10 (11th edition)
   arch12 (z14)     SA22-7832-11 (12th edition)
   arch13 (z15)     SA22-7832-12 (13th edition)
   arch14 (z16)     SA22-7832-13 (14th edition)

   Also based on "z/Architecture: Reference Summary"
   Below is a table showing Z model corresponding revision number:
   arch13 (z15)     SA22-7871-10 (10th edition)
   arch14 (z16)     SA22-7871-11 (11th edition)
*/


LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vxrs_alu( libxsmm_generated_code *io_generated_code,
                                   libxsmm_datatype const  i_datatype,
                                   unsigned int            i_a,
                                   unsigned int            i_b,
                                   unsigned int            i_c,
                                   char                    i_alpha,
                                   char                    i_beta ) {
  /*
    VXRS has 5 possible instructions:
    VFMA  c, a, b -> c = a*b + c        | alpha =  1, beta = 1
    VFMS  c, a, b -> c = a*b - c        | alpha =  1, beta = -1
    VFNMA c, a, b -> c = - ( a*b + c )  | alpha = -1, beta = -1
    VFNMS c, a, b -> c = -( a*b - c)    | alpha = -1, beta = 1
    VFM    c, a, b -> c = a*b           | alpha = -1, beta = 0
  */
  unsigned int l_eletype, l_rxb;

  /* If beta is zero we only suport positive multiplication */
  if ( ( i_beta == 0 ) && ( i_alpha != 1) ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  } else if ( i_alpha == 0 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }

  l_eletype = libxsmm_s390x_instr_eletype( io_generated_code, i_datatype );

  if ( i_beta != 0 ) {
    unsigned long l_op;

    switch ( i_datatype ) {
      case LIBXSMM_DATATYPE_F32:
      case LIBXSMM_DATATYPE_F64: {
        l_op = LIBXSMM_S390X_INSTR_VFMS;
      } break;
      default: {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
        return;
      }
    }

    l_op += ( 0x01 * ( i_beta + 1 ) / 2 ) + ( 0x10 * ( 1 - i_alpha ) / 2 );
    l_rxb = libxsmm_s390x_instr_rxb( io_generated_code, i_c, i_a, i_b, i_c );
    libxsmm_s390x_instr_7( io_generated_code, l_op, i_c, i_a, i_b, l_eletype, 0, i_c, l_rxb );
  } else {
    l_rxb = libxsmm_s390x_instr_rxb( io_generated_code, i_c, i_a, i_b, l_eletype );
    libxsmm_s390x_instr_7( io_generated_code, LIBXSMM_S390X_INSTR_VFM, i_c, i_a, i_b, 0, 0, l_eletype, l_rxb );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_store_mult( libxsmm_generated_code *io_generated_code,
                                         libxsmm_s390x_reg      *io_reg_tracker,
                                         unsigned int            i_ptr,
                                         unsigned int            i_n,
                                         long int                i_offset,
                                         unsigned int            o_vec ) {
  unsigned int l_vec = o_vec + i_n - 1;
  unsigned int const l_align = LIBXSMM_S390X_ALIGN_QUAD;
  unsigned int l_ptr = i_ptr;
  long int l_offset = i_offset;
  unsigned int l_rxb;

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    l_ptr = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_ptr, l_ptr, i_offset );
    l_offset = 0;
  }

  l_rxb = libxsmm_s390x_instr_rxb( io_generated_code, o_vec, l_vec, l_ptr, l_align );
  libxsmm_s390x_instr_6( io_generated_code, LIBXSMM_S390X_INSTR_VSTM, o_vec, l_vec, l_ptr, l_offset, l_align, l_rxb );

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_ptr );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_store_part( libxsmm_generated_code *io_generated_code,
                                         libxsmm_s390x_reg      *io_reg_tracker,
                                         libxsmm_datatype const  i_datatype,
                                         unsigned int            i_ptr,
                                         unsigned int            i_len,
                                         long int                i_offset,
                                         unsigned int            o_vec ) {
  unsigned int l_len = i_len * libxsmm_s390x_bytes( io_generated_code, i_datatype );
  unsigned int l_ptr = i_ptr;
  long int l_offset = i_offset;
  unsigned int l_rxb;

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    l_ptr = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_ptr, l_ptr, i_offset );
    l_offset = 0;
  }

  l_rxb = libxsmm_s390x_instr_rxb( io_generated_code, 0, l_len, l_ptr, o_vec );
  libxsmm_s390x_instr_5( io_generated_code, LIBXSMM_S390X_INSTR_VSTRL, l_len, l_ptr, l_offset, o_vec, l_rxb );

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_ptr );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_load_mult( libxsmm_generated_code *io_generated_code,
                                        libxsmm_s390x_reg      *io_reg_tracker,
                                        unsigned int            i_ptr,
                                        unsigned int            i_n,
                                        long int                i_offset,
                                        unsigned int            o_vec ) {
  unsigned int l_vec = o_vec + i_n - 1;
  unsigned int const l_align = LIBXSMM_S390X_ALIGN_QUAD;
  unsigned int l_ptr = i_ptr;
  long int l_offset = i_offset;
  unsigned int l_rxb;

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    l_ptr = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_ptr, l_ptr, i_offset );
    l_offset = 0;
  }

  l_rxb = libxsmm_s390x_instr_rxb( io_generated_code, o_vec, l_vec, l_ptr, l_align );
  libxsmm_s390x_instr_6( io_generated_code, LIBXSMM_S390X_INSTR_VLM, o_vec, l_vec, l_ptr, l_offset, l_align, l_rxb );

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_ptr );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_load_part( libxsmm_generated_code *io_generated_code,
                                        libxsmm_s390x_reg      *io_reg_tracker,
                                        libxsmm_datatype const  i_datatype,
                                        unsigned int            i_ptr,
                                        unsigned int            i_len,
                                        long int                i_offset,
                                        unsigned int            o_vec ) {
  unsigned int l_len = i_len * libxsmm_s390x_bytes( io_generated_code, i_datatype );
  unsigned int l_ptr = i_ptr;
  long int l_offset = i_offset;
  unsigned int l_rxb;

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    l_ptr = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_ptr, l_ptr, i_offset );
    l_offset = 0;
  }

  l_rxb = libxsmm_s390x_instr_rxb( io_generated_code, 0, l_len, l_ptr, o_vec );
  libxsmm_s390x_instr_5( io_generated_code, LIBXSMM_S390X_INSTR_VLRL, l_len, l_ptr, l_offset, o_vec, l_rxb );

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_ptr );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_bcast( libxsmm_generated_code *io_generated_code,
                                    libxsmm_datatype const  i_datatype,
                                    unsigned int            i_src,
                                    unsigned int            i_ele,
                                    unsigned int            i_dst ) {
  unsigned int l_eletype = libxsmm_s390x_instr_eletype( io_generated_code, i_datatype );
  unsigned int l_rxb = libxsmm_s390x_instr_rxb( io_generated_code, i_dst, i_src, 0, l_eletype );
  libxsmm_s390x_instr_5( io_generated_code, LIBXSMM_S390X_INSTR_VREP, i_dst, i_src, i_ele, l_eletype, l_rxb );

}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_load_bcast( libxsmm_generated_code *io_generated_code,
                                         libxsmm_s390x_reg      *io_reg_tracker,
                                         libxsmm_datatype const  i_datatype,
                                         unsigned int            i_ptr,
                                         long int                i_offset,
                                         unsigned int            o_vec ) {
  libxsmm_s390x_instr_vec_load_bcast_idx( io_generated_code, io_reg_tracker, i_datatype, 0, i_ptr, i_offset, o_vec );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_load_bcast_idx( libxsmm_generated_code *io_generated_code,
                                             libxsmm_s390x_reg      *io_reg_tracker,
                                             libxsmm_datatype const  i_datatype,
                                             unsigned int            i_idxptr,
                                             unsigned int            i_baseptr,
                                             long int                i_offset,
                                             unsigned int            o_vec ) {
  unsigned int l_eletype = libxsmm_s390x_instr_eletype( io_generated_code, i_datatype );
  unsigned int l_idxptr = i_idxptr;
  long int l_offset = i_offset;
  unsigned int l_rxb;

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    l_idxptr = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_idxptr, l_idxptr, i_offset );
    l_offset = 0;
  }

  l_rxb = libxsmm_s390x_instr_rxb( io_generated_code, o_vec, l_idxptr, i_baseptr, l_eletype );
  libxsmm_s390x_instr_6( io_generated_code, LIBXSMM_S390X_INSTR_VLREP, o_vec, l_idxptr, i_baseptr, l_offset, l_eletype, l_rxb );

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_idxptr );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_store( libxsmm_generated_code *io_generated_code,
                                    libxsmm_s390x_reg      *io_reg_tracker,
                                    unsigned int            i_ptr,
                                    long int                i_offset,
                                    unsigned int            i_vec ) {
  libxsmm_s390x_instr_vec_store_idx( io_generated_code, io_reg_tracker, 0, i_ptr, i_offset, i_vec );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_store_idx( libxsmm_generated_code *io_generated_code,
                                        libxsmm_s390x_reg      *io_reg_tracker,
                                        unsigned int            i_idxptr,
                                        unsigned int            i_baseptr,
                                        long int                i_offset,
                                        unsigned int            i_vec ) {
  unsigned int l_idxptr = i_idxptr;
  unsigned int const l_align = LIBXSMM_S390X_ALIGN_QUAD;
  long int l_offset = i_offset;
  unsigned int l_rxb;

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    l_idxptr = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_idxptr, l_idxptr, i_offset );
    l_offset = 0;
  }

  l_rxb = libxsmm_s390x_instr_rxb( io_generated_code, i_vec, l_idxptr, i_baseptr, l_align );
  libxsmm_s390x_instr_6( io_generated_code, LIBXSMM_S390X_INSTR_VST, i_vec, l_idxptr, i_baseptr, l_offset, l_align, l_rxb );

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_idxptr );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_load( libxsmm_generated_code *io_generated_code,
                                   libxsmm_s390x_reg      *io_reg_tracker,
                                   unsigned int            i_ptr,
                                   long int                i_offset,
                                   unsigned int            o_vec ) {
  libxsmm_s390x_instr_vec_load_idx( io_generated_code, io_reg_tracker, 0, i_ptr, i_offset, o_vec );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_vec_load_idx( libxsmm_generated_code *io_generated_code,
                                       libxsmm_s390x_reg      *io_reg_tracker,
                                       unsigned int            i_idxptr,
                                       unsigned int            i_baseptr,
                                       long int                i_offset,
                                       unsigned int            o_vec ) {
  unsigned int l_idxptr = i_idxptr;
  unsigned int const l_align = LIBXSMM_S390X_ALIGN_QUAD;
  long int l_offset = i_offset;
  unsigned int l_rxb;

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    l_idxptr = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_idxptr, l_idxptr, i_offset );
    l_offset = 0;
  }

  l_rxb = libxsmm_s390x_instr_rxb( io_generated_code, o_vec, l_idxptr, i_baseptr, l_align );
  libxsmm_s390x_instr_6( io_generated_code, LIBXSMM_S390X_INSTR_VL, o_vec, l_idxptr, i_baseptr, l_offset, l_align, l_rxb );

  if ( i_offset > 0x07ff || i_offset <= -0x07ff ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_idxptr );
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_s390x_instr_eletype( libxsmm_generated_code *io_generated_code,
                                          libxsmm_datatype const  i_datatype ) {
  unsigned int o_eletype;
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_BF16:
    case LIBXSMM_DATATYPE_F16: {
      o_eletype = LIBXSMM_S390X_TYPE_H;
    } break;
    case LIBXSMM_DATATYPE_F32: {
      o_eletype = LIBXSMM_S390X_TYPE_W;
    } break;
    case LIBXSMM_DATATYPE_F64: {
      o_eletype = LIBXSMM_S390X_TYPE_D;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      return 0xffffffff;
    }
  }
  return o_eletype;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_s390x_instr_rxb( libxsmm_generated_code *io_generated_code,
                                      unsigned int            i_0,
                                      unsigned int            i_1,
                                      unsigned int            i_2,
                                      unsigned int            i_3 ) {
  return (unsigned int)( ( ( 0x10 & i_0 ) >> 1 ) +
                         ( ( 0x10 & i_1 ) >> 2 ) +
                         ( ( 0x10 & i_2 ) >> 3 )  +
                         ( ( 0x10 & i_3 ) >> 4 ) );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_gpr_load( libxsmm_generated_code *io_generated_code,
                                   libxsmm_s390x_reg      *io_reg_tracker,
                                   unsigned int            i_ptr,
                                   long int                i_offset,
                                   unsigned int            o_dst ) {
  libxsmm_s390x_instr_gpr_load_idx( io_generated_code, io_reg_tracker, 0, i_ptr, i_offset, o_dst );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_gpr_load_idx( libxsmm_generated_code *io_generated_code,
                                       libxsmm_s390x_reg      *io_reg_tracker,
                                       unsigned int            i_idxptr,
                                       unsigned int            i_baseptr,
                                       long int                i_offset,
                                       unsigned int            o_dst ) {
  unsigned int l_idxptr = i_idxptr;
  unsigned int l_offset_l, l_offset_h;
  long int l_offset = i_offset;

  if ( i_offset > 0x07ffff || i_offset <= -0x07ffff ) {
    l_idxptr = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );
    libxsmm_s390x_instr_gpr_add_value( io_generated_code, i_idxptr, l_idxptr, i_offset );
    l_offset = 0;
  }

  if ( l_offset ) {
    l_offset_l = (unsigned int)( 0xfff & l_offset );
    l_offset_h = (unsigned int)( 0xff & (l_offset >> 12 ) );
  } else {
    l_offset_l = 0;
    l_offset_h = 0;
  }

  libxsmm_s390x_instr_5( io_generated_code, LIBXSMM_S390X_INSTR_LG, o_dst, l_idxptr, i_baseptr, l_offset_l, l_offset_h );

  if ( i_offset > 0x07ffff || i_offset <= -0x07ffff ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, l_idxptr );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_gpr_add_value( libxsmm_generated_code *io_generated_code,
                                        unsigned int            i_src,
                                        unsigned int            i_dst,
                                        long int                i_value ) {
  /* Set value */
  if ( i_src == 0 ) {
    libxsmm_s390x_instr_gpr_set_value( io_generated_code, i_dst, i_value );
  /* if source and dest are different, more steps are required */
  } else if ( i_src != i_dst ) {
    /* Load register */
    if ( i_value == 0 ) {
      libxsmm_s390x_instr_gpr_copy( io_generated_code, i_src, i_dst );
    /* Add logcial immediate */
    } else if ( i_value <= 0x7fff && i_value > -0x7fff )  {
        libxsmm_s390x_instr_3( io_generated_code, LIBXSMM_S390X_INSTR_AGHIK, i_dst, i_src, i_value);
    } else if ( i_value <= 0x7fffffff && i_value > -0x7fffffff ) {
      libxsmm_s390x_instr_2( io_generated_code, LIBXSMM_S390X_INSTR_LGR, i_dst, i_src );
      libxsmm_s390x_instr_2( io_generated_code, LIBXSMM_S390X_INSTR_AGFI, i_dst, i_value );
    }
  /* if source and dest are the same, simpler commands can be used */
  } else {
    if ( i_value == 0 ) {
      return;
    } else if ( i_value <= 0x7fff && i_value > -0x7fff ) {
      libxsmm_s390x_instr_2( io_generated_code, LIBXSMM_S390X_INSTR_AGHI, i_dst, i_value );
    } else if ( i_value <= 0x7fffffff && i_value > -0x7fffffff ) {
      libxsmm_s390x_instr_2( io_generated_code, LIBXSMM_S390X_INSTR_AGFI, i_dst, i_value );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_gpr_copy( libxsmm_generated_code *io_generated_code,
                                   unsigned int            i_src,
                                   unsigned int            i_dst ) {
  libxsmm_s390x_instr_2( io_generated_code, LIBXSMM_S390X_INSTR_LGR, i_dst, i_src );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_gpr_set_value( libxsmm_generated_code *io_generated_code,
                                        unsigned int            i_gpr,
                                        long int                i_value ) {
  unsigned int l_value = (unsigned int)( i_value & 0xfffffff );
  libxsmm_s390x_instr_2( io_generated_code, LIBXSMM_S390X_INSTR_LGFI, i_gpr, l_value );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_gpr_imm64( libxsmm_generated_code *io_generated_code,
                                    unsigned int            i_gpr,
                                    unsigned long int       i_value ) {
  unsigned int l_high = 0xffffffff & ( i_value >> 32 );
  unsigned int l_low = 0xffffffff & i_value ;

  libxsmm_s390x_instr_2( io_generated_code, LIBXSMM_S390X_INSTR_LGFI, i_gpr, l_high );
  libxsmm_s390x_instr_5( io_generated_code, LIBXSMM_S390X_INSTR_RISBGN, i_gpr, i_gpr, 0, 0x80 + 32, 32);
  libxsmm_s390x_instr_2( io_generated_code, LIBXSMM_S390X_INSTR_ALGFI, i_gpr, l_low );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_ptr_reg_alloc( libxsmm_generated_code *io_generated_code,
                                  libxsmm_s390x_reg      *io_reg_tracker,
                                  unsigned int            i_ptr,
                                  unsigned int            i_n,
                                  unsigned int            i_ld,
                                  unsigned int            i_max_add,
                                  unsigned int           *o_ptr,
                                  long                   *o_offset ) {
  unsigned int i, j;
  const long l_shift = 0x07f0;
  o_ptr[0] = i_ptr;
  o_offset[0] = 0;

  for ( i = 1; i < i_n ; ++i ) {
    char l_new_required = 1;

    for ( j = 0; j < i && l_new_required ; ++j ) {
      long l_rel_offset = o_offset[j] + (i - j)*i_ld;
      long l_max_offset = l_rel_offset + i_max_add;

      if ( l_max_offset < l_shift ) {
        o_ptr[i] = o_ptr[j];
        o_offset[i] = l_rel_offset;
        l_new_required = 0;
      }
    }

    if ( l_new_required ) {
      unsigned int l_reg = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR );

      if ( 0xffffffff == l_reg ) {
        long l_delta = o_offset[0] + i*i_ld;
        o_ptr[i] = o_ptr[0];
        o_offset[i] = l_delta;
      } else {
        long l_delta = o_offset[0] + i*i_ld + l_shift;
        o_ptr[i] = l_reg;
        o_offset[i] = -l_shift;
        libxsmm_s390x_instr_gpr_add_value( io_generated_code, o_ptr[0], o_ptr[i], l_delta );
      }
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_ptr_reg_dealloc( libxsmm_generated_code *io_generated_code,
                                    libxsmm_s390x_reg      *io_reg_tracker,
                                    unsigned int           *i_ptr,
                                    unsigned int            i_n,
                                    char                    i_skip0 ) {
  unsigned int i, l_skip_ptr = 0;

  if ( i_skip0 ) {
    l_skip_ptr = i_ptr[0];
  } else {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, i_ptr[0] );
  }

  for ( i = 1; i < i_n ; ++i ) {
    if ( ( i_skip0 && l_skip_ptr != i_ptr[i] ) || !i_skip0 ) {
      libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, i_ptr[i] );
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_register_jump_label( libxsmm_generated_code     *io_generated_code,
                                              libxsmm_loop_label_tracker *io_loop_label_tracker ) {
  /* check if we still have label we can jump to */
  if ( io_loop_label_tracker->label_count == 512 ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_EXCEED_JMPLBL );
    return;
  }

  if ( io_generated_code->code_type > 1 ) {
    int l_lab = io_loop_label_tracker->label_count;
    io_loop_label_tracker->label_count++;
    io_loop_label_tracker->label_address[l_lab] = io_generated_code->code_size;
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_branch_count_jump_label( libxsmm_generated_code     *io_generated_code,
                                                  unsigned int                i_gpr,
                                                  libxsmm_loop_label_tracker *io_loop_label_tracker ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned int l_lab = --io_loop_label_tracker->label_count;
    unsigned int l_b_dst = io_loop_label_tracker->label_address[l_lab];
    unsigned int l_code_head = io_generated_code->code_size;

    /* branch immediate _half words_ */
    unsigned int l_b_imm = 0xffff & (unsigned int)( ( (int)l_b_dst - (int)l_code_head ) / 2 );

    /* branch on count */
    libxsmm_s390x_instr_2( io_generated_code, LIBXSMM_S390X_INSTR_BRCTG, i_gpr, l_b_imm );
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_unpack_args( libxsmm_generated_code *io_generated_code,
                                      libxsmm_s390x_reg      *io_reg_tracker ) {
  int l_offset_ptr_a = (int)sizeof(libxsmm_matrix_op_arg);
  int l_offset_ptr_b = (int)(sizeof(libxsmm_matrix_op_arg) + sizeof(libxsmm_matrix_arg));
  int l_offset_ptr_c = (int)(sizeof(libxsmm_matrix_op_arg) + 2*sizeof(libxsmm_matrix_arg));

  libxsmm_s390x_instr_gpr_copy( io_generated_code, LIBXSMM_S390X_GPR_ARG0, 1 );
  libxsmm_s390x_instr_gpr_load( io_generated_code, io_reg_tracker, 1, l_offset_ptr_a, LIBXSMM_S390X_GPR_ARG0 );
  libxsmm_s390x_instr_gpr_load( io_generated_code, io_reg_tracker, 1, l_offset_ptr_b, LIBXSMM_S390X_GPR_ARG1 );
  libxsmm_s390x_instr_gpr_load( io_generated_code, io_reg_tracker, 1, l_offset_ptr_c, LIBXSMM_S390X_GPR_ARG2 );

  libxsmm_s390x_reg_set( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, LIBXSMM_S390X_GPR_ARG0, LIBXSMM_S390X_REG_USED );
  libxsmm_s390x_reg_set( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, LIBXSMM_S390X_GPR_ARG1, LIBXSMM_S390X_REG_USED );
  libxsmm_s390x_reg_set( io_generated_code, io_reg_tracker, LIBXSMM_S390X_GPR, LIBXSMM_S390X_GPR_ARG2, LIBXSMM_S390X_REG_USED );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_open_stack( libxsmm_generated_code *io_generated_code ) {
  /* Based on "ELF Application Binary Interface s390x Supplement: v1.6.1"
   */
  unsigned int i, l_dl, l_dh;

  /* Increment the stack pointer */
  l_dl = (unsigned int)(0x0fff & ( -LIBXSMM_S390X_STACK_SIZE ));
  l_dh = (unsigned int)(0xff & ( ( -LIBXSMM_S390X_STACK_SIZE ) >> 12 ));
  libxsmm_s390x_instr_5( io_generated_code, LIBXSMM_S390X_INSTR_LAY, LIBXSMM_S390X_GPR_SP, 0, LIBXSMM_S390X_GPR_SP, l_dl, l_dh );

  /* Store non-volatile GPR */
  libxsmm_s390x_instr_5( io_generated_code, LIBXSMM_S390X_INSTR_STMG, 6, 14, LIBXSMM_S390X_GPR_SP, 8, 0 );

  /* Store non-volatile FPR */
  for ( i = 0 ; i < 8 ; ++i ) {
    unsigned int l_fpr = 8 + i;
    libxsmm_s390x_instr_4( io_generated_code, LIBXSMM_S390X_INSTR_STD, l_fpr, 0, LIBXSMM_S390X_GPR_SP, 80 + i*8 );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_collapse_stack( libxsmm_generated_code *io_generated_code ) {
  /* Based on "ELF Application Binary Interface s390x Supplement: v1.6.1"
   */
  unsigned int i, l_dl, l_dh;

  /* Restore non-volatile GPR */
  libxsmm_s390x_instr_5( io_generated_code, LIBXSMM_S390X_INSTR_LMG, 6, 14, LIBXSMM_S390X_GPR_SP, 8, 0 );

  /* Restore non-volatile FPR */
  for ( i = 0 ; i < 8 ; ++i ) {
    unsigned int l_fpr = 8 + i;
    libxsmm_s390x_instr_4( io_generated_code, LIBXSMM_S390X_INSTR_LD, l_fpr, 0, LIBXSMM_S390X_GPR_SP, 80 + i*8 );
  }

  /* Restore the stack pointer */
  l_dl = (unsigned int)(0x0fff & LIBXSMM_S390X_STACK_SIZE );
  l_dh = (unsigned int)(0xff & ( LIBXSMM_S390X_STACK_SIZE >> 12 ));
  libxsmm_s390x_instr_5( io_generated_code, LIBXSMM_S390X_INSTR_LAY, LIBXSMM_S390X_GPR_SP, 0, LIBXSMM_S390X_GPR_SP, l_dl, l_dh );

  /* Return */
  libxsmm_s390x_instr_return( io_generated_code );

  return;
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_return( libxsmm_generated_code *io_generated_code ) {
  unsigned int l_head = io_generated_code->code_size;
  unsigned short *l_code = (unsigned short*)( (unsigned char *)io_generated_code->generated_code + l_head );
  *l_code = (unsigned short)(0xffff & LIBXSMM_S390X_INSTR_RETURN );
  io_generated_code->code_size += 2;
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_nop( libxsmm_generated_code *io_generated_code ) {
  unsigned int l_head = io_generated_code->code_size;
  unsigned int *l_code = (unsigned int *)( (unsigned char *)io_generated_code->generated_code + l_head);
  *l_code = (unsigned int)(0xffffffff & LIBXSMM_S390X_INSTR_NOP );
  io_generated_code->code_size += 4;
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_nopr( libxsmm_generated_code *io_generated_code ) {
  unsigned int l_head = io_generated_code->code_size;
  unsigned short *l_code = (unsigned short*)( (unsigned char *)io_generated_code->generated_code + l_head );
  *l_code = (unsigned short)(0xffff & LIBXSMM_S390X_INSTR_NOPR );
  io_generated_code->code_size += 2;
}

LIBXSMM_API_INTERN
libxsmm_s390x_reg libxsmm_s390x_reg_init( libxsmm_generated_code *io_generated_code ) {
  unsigned int l_gpr_res[] = LIBXSMM_S390X_RESV_GPR;
  unsigned int l_ngpr, l_nfpr, l_nvr;
  libxsmm_s390x_reg_stack *l_gpr;
  unsigned int *l_fpr, *l_vr;
  libxsmm_s390x_reg o_reg = {0, NULL, 0, NULL, 0, NULL };
  unsigned int i;

  switch( io_generated_code->arch ) {
    case LIBXSMM_S390X_ARCH11: {
      l_ngpr = LIBXSMM_S390X_ARCH11_GPR;
      l_nfpr = LIBXSMM_S390X_ARCH11_FPR;
      l_nvr = LIBXSMM_S390X_ARCH11_VR;
    } break;
    case LIBXSMM_S390X_ARCH12: {
      l_ngpr = LIBXSMM_S390X_ARCH12_GPR;
      l_nfpr = LIBXSMM_S390X_ARCH12_FPR;
      l_nvr = LIBXSMM_S390X_ARCH12_VR;
    } break;
    case LIBXSMM_S390X_ARCH13: {
      l_ngpr = LIBXSMM_S390X_ARCH13_GPR;
      l_nfpr = LIBXSMM_S390X_ARCH13_FPR;
      l_nvr = LIBXSMM_S390X_ARCH13_VR;
    } break;
    case LIBXSMM_S390X_ARCH14: {
      l_ngpr = LIBXSMM_S390X_ARCH14_GPR;
      l_nfpr = LIBXSMM_S390X_ARCH14_FPR;
      l_nvr = LIBXSMM_S390X_ARCH14_VR;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_ARCH );
      return o_reg;
    }
  }
  l_gpr = libxsmm_s390x_reg_stack_init(l_ngpr, LIBXSMM_S390X_GPR, l_gpr_res, sizeof(l_gpr_res) / sizeof(l_gpr_res[0]) );
  l_fpr = malloc(sizeof(unsigned int)*l_nfpr);
  l_vr = malloc(sizeof(unsigned int)*l_nvr);

  for ( i = 0 ; i < l_nfpr ; ++i ) {
    l_fpr[i] = LIBXSMM_S390X_REG_FREE;
  }
  for ( i = 0 ; i < l_nvr ; ++i ) {
    l_vr[i] = LIBXSMM_S390X_REG_FREE;
  }

  o_reg.ngpr = l_ngpr;
  o_reg.gpr = l_gpr;
  o_reg.nfpr = l_nfpr;
  o_reg.fpr = l_fpr;
  o_reg.nvr = l_nvr;
  o_reg.vr = l_vr;
  return o_reg;
}

LIBXSMM_API_INTERN
libxsmm_s390x_reg_stack* libxsmm_s390x_reg_stack_init( unsigned int            i_n,
                                                       libxsmm_s390x_reg_type  i_reg_type,
                                                       unsigned int           *i_resv,
                                                       unsigned int            i_nresv ) {
  libxsmm_s390x_reg_util *l_util = malloc(i_n*sizeof(enum libxsmm_s390x_reg_util));
  libxsmm_s390x_reg_type *l_type = malloc(i_n*sizeof(enum libxsmm_s390x_reg_type));
  libxsmm_s390x_reg_node *l_stack = malloc(i_n*sizeof(libxsmm_s390x_reg_node));
  libxsmm_s390x_reg_stack *out = malloc(sizeof(libxsmm_s390x_reg_stack));
  libxsmm_s390x_reg_node *l_prev = NULL;
  unsigned int i, j;

  for ( i = 0; i < i_n ; ++i ) {
    libxsmm_s390x_reg_node *l_node = &l_stack[i];
    enum libxsmm_s390x_reg_util l_ru = LIBXSMM_S390X_REG_FREE;

    for ( j = 0; j < i_nresv; ++j ) {
      l_ru = ( i_resv[j] == i ) ? LIBXSMM_S390X_REG_RESV : l_ru;
    }

    l_util[i] = l_ru;
    l_type[i] = i_reg_type;
    l_node->u = &l_util[i];
    l_node->t = &l_type[i];
    l_node->id = i;
    l_node->prev = NULL;
    l_node->next = NULL;
  }

  for ( i = 0 ; i < i_n ; ++i ) {
    libxsmm_s390x_reg_node *l_node = &l_stack[i];
    if ( LIBXSMM_S390X_REG_FREE == *l_node->u ) {
      if ( NULL != l_prev ) {
        l_node->prev = l_prev;
        l_prev->next = l_node;
      }
      l_prev = l_node;
    }
  }

  out->n = i_n;
  out->util = l_util;
  out->type = l_type;
  out->free = libxsmm_s390x_reg_stack_head( l_prev );
  out->stack = l_stack;

  return out;
}

LIBXSMM_API_INTERN
libxsmm_s390x_reg_node* libxsmm_s390x_reg_stack_head( libxsmm_s390x_reg_node *i_reg ) {
  libxsmm_s390x_reg_node *head = i_reg;
  if ( NULL != head ) {
    while ( NULL != head->prev ) {
      head = head->prev;
    }
  }
  return head;
}

LIBXSMM_API_INTERN
libxsmm_s390x_reg_node* libxsmm_s390x_reg_stack_tail( libxsmm_s390x_reg_node *i_reg ) {
  libxsmm_s390x_reg_node *tail = i_reg;
  if ( NULL != tail ) {
    while ( NULL != tail->next ) {
      tail = tail->next;
    }
  }
  return tail;
}

LIBXSMM_API_INTERN
unsigned int libxsmm_s390x_reg_stack_get( libxsmm_s390x_reg_stack *i_stack ) {
  unsigned int out;

  if ( NULL != i_stack->free ) {
    libxsmm_s390x_reg_node *l_free = i_stack->free;
    libxsmm_s390x_reg_node *l_next = i_stack->free->next;

    *l_free->u = LIBXSMM_S390X_REG_USED;
    l_free->next = NULL;
    l_next->prev = NULL;
    i_stack->free = l_next;

    out = l_free->id;
  } else {
    out = 0xffffffff;
  }
  return out;
}

LIBXSMM_API_INTERN
void libxsmm_s390x_reg_stack_free( libxsmm_s390x_reg_stack *i_stack,
                                   unsigned int             i_reg ) {
  libxsmm_s390x_reg_node *l_node = &i_stack->stack[i_reg];

  if ( LIBXSMM_S390X_REG_FREE != *l_node->u ) {
    libxsmm_s390x_reg_node *l_last = libxsmm_s390x_reg_stack_tail( i_stack->free );
    *l_node->u = LIBXSMM_S390X_REG_FREE;
    l_node->next = NULL;

    if ( NULL != l_last ) {
      l_last->next = l_node;
      l_node->prev = l_last;
    } else {
      i_stack->free = l_node;
      l_node->prev = NULL;
    }
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_reg_stack_set( libxsmm_s390x_reg_stack *i_stack,
                                  unsigned int             i_reg,
                                  libxsmm_s390x_reg_util   i_value ) {
  if ( LIBXSMM_S390X_REG_FREE == i_value ) {
    libxsmm_s390x_reg_stack_free( i_stack, i_reg );
  } else {
    libxsmm_s390x_reg_node *l_node = &i_stack->stack[i_reg];

    if ( LIBXSMM_S390X_REG_FREE == *l_node->u ) {
      if ( NULL != l_node->next ) {
        l_node->next->prev = l_node->prev;
      }
      if ( NULL != l_node->prev ) {
        l_node->prev->next = l_node->next;
      }
      if ( l_node == i_stack->free ) {
        i_stack->free = l_node->next;
      }
    }
    l_node->next = NULL;
    l_node->prev = NULL;
    *l_node->u = i_value;
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_reg_destroy( libxsmm_generated_code *io_generated_code,
                                libxsmm_s390x_reg      *i_reg_tracker ) {
  i_reg_tracker->ngpr = 0;
  free(i_reg_tracker->gpr->util);
  free(i_reg_tracker->gpr->type);
  free(i_reg_tracker->gpr->stack);
  free(i_reg_tracker->gpr);
  i_reg_tracker->nfpr = 0;
  free(i_reg_tracker->fpr);
  i_reg_tracker->nvr = 0;
  free(i_reg_tracker->vr);
}

LIBXSMM_API_INTERN
void libxsmm_s390x_reg_alloc( libxsmm_generated_code *io_generated_code,
                              libxsmm_s390x_reg      *io_reg_tracker,
                              libxsmm_s390x_reg_type  i_reg_type,
                              unsigned int            i_n,
                              unsigned int           *o_reg ) {
  unsigned int i;
  for ( i = 0 ; i < i_n ; ++i ) {
    o_reg[i] = libxsmm_s390x_reg_get( io_generated_code, io_reg_tracker, i_reg_type );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_reg_dealloc( libxsmm_generated_code *io_generated_code,
                                libxsmm_s390x_reg      *io_reg_tracker,
                                libxsmm_s390x_reg_type  i_reg_type,
                                unsigned int            i_n,
                                unsigned int           *o_reg ) {
  unsigned int i;
  for ( i = 0 ; i < i_n ; ++i ) {
    libxsmm_s390x_reg_free( io_generated_code, io_reg_tracker, i_reg_type, o_reg[i] );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_reg_alloc_vr_mat( libxsmm_generated_code *io_generated_code,
                                     libxsmm_s390x_reg      *io_reg_tracker,
                                     unsigned int            i_m,
                                     unsigned int            i_n,
                                     unsigned int           *o_reg ) {
  unsigned int l_col;
  for ( l_col = 0 ; l_col < i_n ; ++l_col ) {
    libxsmm_s390x_reg_get_contig( io_generated_code,
                                  io_reg_tracker,
                                  LIBXSMM_S390X_VR,
                                  i_m,
                                  &o_reg[l_col*i_m] );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_reg_get_contig( libxsmm_generated_code *io_generated_code,
                                   libxsmm_s390x_reg      *io_reg_tracker,
                                   libxsmm_s390x_reg_type  i_reg_type,
                                   unsigned int            i_num,
                                   unsigned int           *o_reg ) {
  unsigned int l_alloc;
  int l_nreg, i, j;
  l_alloc = 0;
  i = 0;

  switch(i_reg_type) {
    case LIBXSMM_S390X_FPR: {
      l_nreg = (int)io_reg_tracker->nfpr;
    } break;
    case LIBXSMM_S390X_VR: {
      l_nreg = (int)io_reg_tracker->nvr;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return;
    }
  }

  while ( ( i <= (int)(l_nreg - i_num) ) && !l_alloc ) {
    char l_search = 1;
    for ( j = 0 ; j < (int)i_num && l_search ; ++j ) {
      if ( !libxsmm_s390x_reg_isfree( io_generated_code, io_reg_tracker, i_reg_type, j + i ) ) {
        i += j + 1;
        l_search = 0;
      }
    }
    if ( l_search ) {
      l_alloc = 1;
    }
  }

  for ( j = 0 ; j < (int)i_num && l_alloc ; ++j ) {
    libxsmm_s390x_reg_used( io_generated_code, io_reg_tracker, i_reg_type, j + i);
    o_reg[j] = (unsigned int)(j + i);
  }

  if ( !l_alloc ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return;
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_s390x_reg_get( libxsmm_generated_code *io_generated_code,
                                    libxsmm_s390x_reg      *io_reg_tracker,
                                    libxsmm_s390x_reg_type  i_reg_type ) {
  char l_alloc;
  unsigned int o_reg, i;
  int l_nreg;

  l_alloc = 0;
  o_reg = 0xffffffff;

  switch(i_reg_type) {
    case LIBXSMM_S390X_GPR: {
      return libxsmm_s390x_reg_stack_get( io_reg_tracker->gpr );
    } break;
    case LIBXSMM_S390X_FPR: {
      l_nreg = (int)io_reg_tracker->nfpr;
    } break;
    case LIBXSMM_S390X_VR: {
      l_nreg = (int)io_reg_tracker->nvr;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return o_reg;
    }
  }

  for ( i = 0; i < (unsigned int)l_nreg && !l_alloc ; ++i ) {
    if ( libxsmm_s390x_reg_isfree( io_generated_code, io_reg_tracker, i_reg_type, i) ) {
      libxsmm_s390x_reg_used( io_generated_code, io_reg_tracker, i_reg_type, i );
      l_alloc = 1;
      o_reg = i;
    }
  }

  if ( !l_alloc ) {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    return o_reg;
  }

  return o_reg;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_reg_isfree(libxsmm_generated_code *io_generated_code,
                              libxsmm_s390x_reg      *io_reg_tracker,
                              libxsmm_s390x_reg_type  i_reg_type,
                              unsigned int            i_reg ) {
  char l_isfree = 0;
  switch(i_reg_type) {
    case LIBXSMM_S390X_GPR: {
      l_isfree = ( io_reg_tracker->gpr->util[i_reg] == LIBXSMM_S390X_REG_FREE ) ? 1 : 0;
    } break;
    case LIBXSMM_S390X_FPR: {
      l_isfree = ( ( io_reg_tracker->fpr[i_reg] >= LIBXSMM_S390X_REG_FREE ) &&
                   ( io_reg_tracker->vr[i_reg] >= LIBXSMM_S390X_REG_FREE ) ) ? 1 : 0;
    } break;
    case LIBXSMM_S390X_VR: {
      if ( i_reg >= io_reg_tracker->nfpr ) {
        l_isfree = ( io_reg_tracker->vr[i_reg] == LIBXSMM_S390X_REG_FREE ) ? 1 : 0;
      } else {
        l_isfree = ( ( io_reg_tracker->fpr[i_reg] >= LIBXSMM_S390X_REG_FREE ) &&
                     ( io_reg_tracker->vr[i_reg] >= LIBXSMM_S390X_REG_FREE ) ) ? 1 : 0;
      }
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      return l_isfree;
    }
  }
  return l_isfree;
}

LIBXSMM_API_INTERN
void libxsmm_s390x_reg_free( libxsmm_generated_code *io_generated_code,
                             libxsmm_s390x_reg      *io_reg_tracker,
                             libxsmm_s390x_reg_type  i_reg_type,
                             unsigned int            i_reg ) {
  libxsmm_s390x_reg_set( io_generated_code, io_reg_tracker, i_reg_type, i_reg, LIBXSMM_S390X_REG_FREE );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_reg_used( libxsmm_generated_code *io_generated_code,
                             libxsmm_s390x_reg      *io_reg_tracker,
                             libxsmm_s390x_reg_type  i_reg_type,
                             unsigned int            i_reg ) {
  libxsmm_s390x_reg_set( io_generated_code, io_reg_tracker, i_reg_type, i_reg, LIBXSMM_S390X_REG_USED );
}

LIBXSMM_API_INTERN
void libxsmm_s390x_reg_set( libxsmm_generated_code *io_generated_code,
                            libxsmm_s390x_reg      *io_reg_tracker,
                            libxsmm_s390x_reg_type  i_reg_type,
                            unsigned int            i_reg,
                            libxsmm_s390x_reg_util  i_value ) {
  switch(i_reg_type) {
    case LIBXSMM_S390X_GPR: {
      if ( i_reg < io_reg_tracker->ngpr ) {
        libxsmm_s390x_reg_stack_set( io_reg_tracker->gpr, i_reg, i_value );
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      }
    } break;
    case LIBXSMM_S390X_FPR: {
      if ( i_reg < io_reg_tracker->nfpr ) {
        io_reg_tracker->fpr[i_reg] = i_value;
        io_reg_tracker->vr[i_reg] = i_value;
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      }
    } break;
    case LIBXSMM_S390X_VR: {
      if ( i_reg < io_reg_tracker->nvr ) {
        io_reg_tracker->vr[i_reg] = i_value;
        if ( i_reg < io_reg_tracker->nfpr ) {
          io_reg_tracker->fpr[i_reg] = i_value;
        }
      } else {
        LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
      }
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  }
}

LIBXSMM_API_INTERN
unsigned int libxsmm_s390x_bytes( libxsmm_generated_code *io_generated_code,
                                  const libxsmm_datatype  i_datatype ) {
  unsigned int o_len;
  switch ( i_datatype ) {
    case LIBXSMM_DATATYPE_F16:
    case LIBXSMM_DATATYPE_BF16: {
      o_len = 2;
    } break;
    case LIBXSMM_DATATYPE_F32: {
      o_len = 4;
    } break;
    case LIBXSMM_DATATYPE_F64: {
      o_len = 8;
    } break;
    default: {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_UNSUP_DATATYPE );
      o_len = 0;
    }
  }
  return o_len;
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_append( libxsmm_generated_code *io_generated_code,
                                 unsigned char          *i_op,
                                 char                    i_nbytes ) {
  if ( io_generated_code->code_type > 1 ) {
    unsigned char *l_code = (unsigned char*) io_generated_code->generated_code;
    unsigned int l_code_head = io_generated_code->code_size;
    int i;
    for ( i = 0; i < (int)i_nbytes ; ++i ) {
      l_code[l_code_head + i] = i_op[i];
    }
    io_generated_code->code_size += i_nbytes;
  }
  else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_0( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = i_instr & LIBXSMM_S390X_FMASK;
    unsigned long l_instr = i_instr & ~LIBXSMM_S390X_FMASK;

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_e_form( (unsigned int)l_instr, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_1( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = i_instr & LIBXSMM_S390X_FMASK;
    unsigned long l_instr = i_instr & ~LIBXSMM_S390X_FMASK;

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_I_FORM: {
        l_nbytes = libxsmm_s390x_instr_i_form( (unsigned int)l_instr, (unsigned char)i_0, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_2( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = i_instr & LIBXSMM_S390X_FMASK;
    unsigned long l_instr = i_instr & ~LIBXSMM_S390X_FMASK;

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_IE_FORM: {
        l_nbytes = libxsmm_s390x_instr_ie_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RI_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_ri_a_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RI_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_ri_b_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RI_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_ri_c_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIL_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_ril_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIL_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_ril_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIL_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_ril_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RR_FORM: {
        l_nbytes = libxsmm_s390x_instr_rr_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRE_FORM: {
        l_nbytes = libxsmm_s390x_instr_rre_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_S_FORM: {
        l_nbytes = libxsmm_s390x_instr_s_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned int)i_1, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_G_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_g_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_3( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = i_instr & LIBXSMM_S390X_FMASK;
    unsigned long l_instr = i_instr & ~LIBXSMM_S390X_FMASK;

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_MII_FORM: {
        l_nbytes = libxsmm_s390x_instr_mii_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIE_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIE_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_d_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIE_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_e_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIE_G_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_g_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRD_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrd_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RSI_FORM: {
        l_nbytes = libxsmm_s390x_instr_rsi_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RSL_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rsl_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RSL_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rsl_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SI_FORM: {
        l_nbytes = libxsmm_s390x_instr_si_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SIL_FORM: {
        l_nbytes = libxsmm_s390x_instr_sil_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned int)i_2, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_4( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = i_instr & LIBXSMM_S390X_FMASK;
    unsigned long l_instr = i_instr & ~LIBXSMM_S390X_FMASK;

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_RIE_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIE_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRF_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrf_a_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRF_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrf_b_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRF_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrf_c_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRF_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrf_d_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRF_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrf_e_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RS_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rs_a_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RS_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rs_b_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RX_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rx_a_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RX_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rx_b_form( (unsigned int)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SIY_FORM: {
        l_nbytes = libxsmm_s390x_instr_siy_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SMI_FORM: {
        l_nbytes = libxsmm_s390x_instr_smi_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SSE_FORM: {
        l_nbytes = libxsmm_s390x_instr_sse_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, (unsigned int)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_H_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_h_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned int)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_F_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_f_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_H_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_h_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_K_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_k_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_5( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3,
                            unsigned int            i_4 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = i_instr & LIBXSMM_S390X_FMASK;
    unsigned long l_instr = i_instr & ~LIBXSMM_S390X_FMASK;

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_RIE_F_FORM: {
        l_nbytes = libxsmm_s390x_instr_rie_f_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RIS_FORM: {
        l_nbytes = libxsmm_s390x_instr_ris_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RRS_FORM: {
        l_nbytes = libxsmm_s390x_instr_rrs_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RSY_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rsy_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RSY_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rsy_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RXE_FORM: {
        l_nbytes = libxsmm_s390x_instr_rxe_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RXF_FORM: {
        l_nbytes = libxsmm_s390x_instr_rxf_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RXY_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_rxy_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_RXY_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_rxy_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SS_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned int)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SS_F_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_f_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned int)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SSF_FORM: {
        l_nbytes = libxsmm_s390x_instr_ssf_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned int)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_I_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_i_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_I_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_i_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_J_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_j_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRS_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrs_d_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VSI_FORM: {
        l_nbytes = libxsmm_s390x_instr_vsi_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned char)i_4, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }

    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_6( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3,
                            unsigned int            i_4,
                            unsigned int            i_5 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = i_instr & LIBXSMM_S390X_FMASK;
    unsigned long l_instr = i_instr & ~LIBXSMM_S390X_FMASK;

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_SS_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned int)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SS_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned int)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SS_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_d_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned int)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_SS_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_ss_e_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned int)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_d_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_e_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned int)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_F_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_f_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRI_G_FORM: {
        l_nbytes = libxsmm_s390x_instr_vri_g_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRS_A_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrs_a_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRS_B_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrs_b_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRS_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrs_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRV_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrv_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRX_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrx_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned int)i_3, (unsigned char)i_4, (unsigned char)i_5, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }
    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}

LIBXSMM_API_INTERN
void libxsmm_s390x_instr_7( libxsmm_generated_code *io_generated_code,
                            unsigned long           i_instr,
                            unsigned int            i_0,
                            unsigned int            i_1,
                            unsigned int            i_2,
                            unsigned int            i_3,
                            unsigned int            i_4,
                            unsigned int            i_5,
                            unsigned int            i_6 ) {
  if ( io_generated_code->code_type > 1 ) {
    char l_nbytes;
    unsigned char l_out[8];

    unsigned long l_fid = i_instr & LIBXSMM_S390X_FMASK;
    unsigned long l_instr = i_instr & ~LIBXSMM_S390X_FMASK;

    switch( l_fid ) {
      case LIBXSMM_S390X_FORM_VRR_C_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_c_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_D_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_d_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, l_out);
      } break;
      case LIBXSMM_S390X_FORM_VRR_E_FORM: {
        l_nbytes = libxsmm_s390x_instr_vrr_e_form( (unsigned long)l_instr, (unsigned char)i_0, (unsigned char)i_1, (unsigned char)i_2, (unsigned char)i_3, (unsigned char)i_4, (unsigned char)i_5, (unsigned char)i_6, l_out);
      } break;
      default: {
        l_nbytes = 0;
      }
    }

    if ( l_nbytes > 0 ) {
      libxsmm_s390x_instr_append( io_generated_code, l_out, l_nbytes );
    } else {
      LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
    }
  } else {
    LIBXSMM_HANDLE_ERROR( io_generated_code, LIBXSMM_ERR_GENERAL );
  }
}


/* All code below here is auto-generated */

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_e_form(unsigned int instr, unsigned char *output) {
unsigned int opcode = (0xffff & instr);
output[0] = 0xff & (opcode >> 8);
output[1] = 0xff & (opcode >> 0);
return 2;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_i_form(unsigned int instr, unsigned char i, unsigned char *output) {
unsigned int opcode = (0xffff & instr);
opcode += (unsigned int)(0xff & i);
output[0] = 0xff & (opcode >> 8);
output[1] = 0xff & (opcode >> 0);
return 2;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ie_form(unsigned int instr, unsigned char i1,unsigned char i2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & i1) << 4;
opcode += (unsigned int)(0xf & i2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_mii_form(unsigned long instr, unsigned char m1,unsigned int ri2,unsigned int ri3, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & m1) << 36;
opcode += (unsigned long)(0xfff & ri2) << 24;
opcode += (unsigned long)(0xffffff & ri3);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ri_a_form(unsigned int instr, unsigned char r1,unsigned int i2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xffff & i2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ri_b_form(unsigned int instr, unsigned char r1,unsigned int ri1, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xffff & ri1);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ri_c_form(unsigned int instr, unsigned char m1,unsigned int ri2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & m1) << 20;
opcode += (unsigned int)(0xffff & ri2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_a_form(unsigned long instr, unsigned char r1,unsigned int i2,unsigned char m3, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xffff & i2) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_b_form(unsigned long instr, unsigned char r1,unsigned char r2,unsigned int ri4,unsigned char m3, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r2) << 32;
opcode += (unsigned long)(0xffff & ri4) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_c_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned int ri4,unsigned char i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & m3) << 32;
opcode += (unsigned long)(0xffff & ri4) << 16;
opcode += (unsigned long)(0xff & i2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_d_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned int i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xffff & i2) << 16;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_e_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned int ri2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xffff & ri2) << 16;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_f_form(unsigned long instr, unsigned char r1,unsigned char r2,unsigned char i3,unsigned char i4,unsigned char i5, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r2) << 32;
opcode += (unsigned long)(0xff & i3) << 24;
opcode += (unsigned long)(0xff & i4) << 16;
opcode += (unsigned long)(0xff & i5) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rie_g_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned int i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & m3) << 32;
opcode += (unsigned long)(0xffff & i2) << 16;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ril_a_form(unsigned long instr, unsigned char r1,unsigned int i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xffffffff & i2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ril_b_form(unsigned long instr, unsigned char r1,unsigned int ri2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xffffffff & ri2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ril_c_form(unsigned long instr, unsigned char m1,unsigned int ri2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & m1) << 36;
opcode += (unsigned long)(0xffffffff & ri2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ris_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned char b4,unsigned int d4,unsigned char i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & m3) << 32;
opcode += (unsigned long)(0xf & b4) << 28;
opcode += (unsigned long)(0xfff & d4) << 16;
opcode += (unsigned long)(0xff & i2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rr_form(unsigned int instr, unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffff & instr);
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 8);
output[1] = 0xff & (opcode >> 0);
return 2;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrd_form(unsigned int instr, unsigned char r1,unsigned char r3,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 12;
opcode += (unsigned int)(0xf & r3) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rre_form(unsigned int instr, unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_a_form(unsigned int instr, unsigned char r3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r3) << 12;
opcode += (unsigned int)(0xf & m4) << 8;
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_b_form(unsigned int instr, unsigned char r3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r3) << 12;
opcode += (unsigned int)(0xf & m4) << 8;
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_c_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & m3) << 12;
opcode += (unsigned int)(0xf & m4) << 8;
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_d_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & m3) << 12;
opcode += (unsigned int)(0xf & m4) << 8;
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrf_e_form(unsigned int instr, unsigned char m3,unsigned char m4,unsigned char r1,unsigned char r2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & m3) << 12;
opcode += (unsigned int)(0xf & m4) << 8;
opcode += (unsigned int)(0xf & r1) << 4;
opcode += (unsigned int)(0xf & r2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rrs_form(unsigned long instr, unsigned char r1,unsigned char r2,unsigned char b4,unsigned int d4,unsigned char m4, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r2) << 32;
opcode += (unsigned long)(0xf & b4) << 28;
opcode += (unsigned long)(0xfff & d4) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rs_a_form(unsigned int instr, unsigned char r1,unsigned char r3,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xf & r3) << 16;
opcode += (unsigned int)(0xf & b2) << 12;
opcode += (unsigned int)(0xfff & d2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rs_b_form(unsigned int instr, unsigned char r1,unsigned char m3,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xf & m3) << 16;
opcode += (unsigned int)(0xf & b2) << 12;
opcode += (unsigned int)(0xfff & d2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsi_form(unsigned int instr, unsigned char r1,unsigned char r3,unsigned int ri2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xf & r3) << 16;
opcode += (unsigned int)(0xffff & ri2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsl_a_form(unsigned long instr, unsigned char l1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & l1) << 36;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsl_b_form(unsigned long instr, unsigned char l1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xff & l1) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsy_a_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & dl2) << 16;
opcode += (unsigned long)(0xff & dh2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rsy_b_form(unsigned long instr, unsigned char r1,unsigned char m3,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & m3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & dl2) << 16;
opcode += (unsigned long)(0xff & dh2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rx_a_form(unsigned int instr, unsigned char r1,unsigned char x2,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & r1) << 20;
opcode += (unsigned int)(0xf & x2) << 16;
opcode += (unsigned int)(0xf & b2) << 12;
opcode += (unsigned int)(0xfff & d2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rx_b_form(unsigned int instr, unsigned char m1,unsigned char x2,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & m1) << 20;
opcode += (unsigned int)(0xf & x2) << 16;
opcode += (unsigned int)(0xf & b2) << 12;
opcode += (unsigned int)(0xfff & d2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxe_form(unsigned long instr, unsigned char r1,unsigned char x2,unsigned char b2,unsigned int d2,unsigned char m3, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & x2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxf_form(unsigned long instr, unsigned char r3,unsigned char x2,unsigned char b2,unsigned int d2,unsigned char r1, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r3) << 36;
opcode += (unsigned long)(0xf & x2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & r1) << 12;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxy_a_form(unsigned long instr, unsigned char r1,unsigned char x2,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & x2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & dl2) << 16;
opcode += (unsigned long)(0xff & dh2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_rxy_b_form(unsigned long instr, unsigned char m1,unsigned char x2,unsigned char b2,unsigned int dl2,unsigned char dh2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & m1) << 36;
opcode += (unsigned long)(0xf & x2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & dl2) << 16;
opcode += (unsigned long)(0xff & dh2) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_s_form(unsigned int instr, unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xf & b2) << 12;
opcode += (unsigned int)(0xfff & d2);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_si_form(unsigned int instr, unsigned char i2,unsigned char b1,unsigned int d1, unsigned char *output) {
unsigned int opcode = (0xffffffff & instr);
opcode += (unsigned int)(0xff & i2) << 16;
opcode += (unsigned int)(0xf & b1) << 12;
opcode += (unsigned int)(0xfff & d1);
output[0] = 0xff & (opcode >> 24);
output[1] = 0xff & (opcode >> 16);
output[2] = 0xff & (opcode >> 8);
output[3] = 0xff & (opcode >> 0);
return 4;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_sil_form(unsigned long instr, unsigned char b1,unsigned int d1,unsigned int i2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xffff & i2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_siy_form(unsigned long instr, unsigned char i2,unsigned char b1,unsigned int dl1,unsigned char dh1, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xff & i2) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & dl1) << 16;
opcode += (unsigned long)(0xff & dh1) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_smi_form(unsigned long instr, unsigned char m1,unsigned char b3,unsigned int d3,unsigned int ri2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & m1) << 36;
opcode += (unsigned long)(0xf & b3) << 28;
opcode += (unsigned long)(0xfff & d3) << 16;
opcode += (unsigned long)(0xffff & ri2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_a_form(unsigned long instr, unsigned char l,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xff & l) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_b_form(unsigned long instr, unsigned char l1,unsigned char l2,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & l1) << 36;
opcode += (unsigned long)(0xf & l2) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_c_form(unsigned long instr, unsigned char l1,unsigned char i3,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & l1) << 36;
opcode += (unsigned long)(0xf & i3) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_d_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_e_form(unsigned long instr, unsigned char r1,unsigned char r3,unsigned char b2,unsigned int d2,unsigned char b4,unsigned int d4, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & b4) << 12;
opcode += (unsigned long)(0xfff & d4);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ss_f_form(unsigned long instr, unsigned char l2,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xff & l2) << 32;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_sse_form(unsigned long instr, unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_ssf_form(unsigned long instr, unsigned char r1,unsigned char b1,unsigned int d1,unsigned char b2,unsigned int d2, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & b1) << 28;
opcode += (unsigned long)(0xfff & d1) << 16;
opcode += (unsigned long)(0xf & b2) << 12;
opcode += (unsigned long)(0xfff & d2);
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_a_form(unsigned long instr, unsigned char v1,unsigned int i2,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xffff & i2) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_b_form(unsigned long instr, unsigned char v1,unsigned char i2,unsigned char i3,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xff & i2) << 24;
opcode += (unsigned long)(0xff & i3) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_c_form(unsigned long instr, unsigned char v1,unsigned char v3,unsigned int i2,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v3) << 32;
opcode += (unsigned long)(0xffff & i2) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_d_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char i4,unsigned char m5,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xff & i4) << 16;
opcode += (unsigned long)(0xf & m5) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_e_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned int i3,unsigned char m5,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xfff & i3) << 20;
opcode += (unsigned long)(0xf & m5) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_f_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m5,unsigned char i4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m5) << 20;
opcode += (unsigned long)(0xff & i4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_g_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char i4,unsigned char m5,unsigned char i3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xff & i4) << 24;
opcode += (unsigned long)(0xf & m5) << 20;
opcode += (unsigned long)(0xff & i3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_h_form(unsigned long instr, unsigned char v1,unsigned int i2,unsigned char i3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xffff & i2) << 16;
opcode += (unsigned long)(0xf & i3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vri_i_form(unsigned long instr, unsigned char v1,unsigned char r2,unsigned char m4,unsigned char i3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & r2) << 32;
opcode += (unsigned long)(0xf & m4) << 20;
opcode += (unsigned long)(0xff & i3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_a_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char m5,unsigned char m4,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & m5) << 20;
opcode += (unsigned long)(0xf & m4) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_b_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m5,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m5) << 20;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_c_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m6,unsigned char m5,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m6) << 20;
opcode += (unsigned long)(0xf & m5) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_d_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m5,unsigned char m6,unsigned char v4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m5) << 24;
opcode += (unsigned long)(0xf & m6) << 20;
opcode += (unsigned long)(0xf & v4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_e_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m6,unsigned char m5,unsigned char v4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m6) << 24;
opcode += (unsigned long)(0xf & m5) << 16;
opcode += (unsigned long)(0xf & v4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_f_form(unsigned long instr, unsigned char v1,unsigned char r2,unsigned char r3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & r2) << 32;
opcode += (unsigned long)(0xf & r3) << 28;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_g_form(unsigned long instr, unsigned char v1,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 32;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_h_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 32;
opcode += (unsigned long)(0xf & v2) << 28;
opcode += (unsigned long)(0xf & m3) << 20;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_i_form(unsigned long instr, unsigned char r1,unsigned char v2,unsigned char m3,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & m3) << 20;
opcode += (unsigned long)(0xf & m4) << 16;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_j_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char v3,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & v3) << 28;
opcode += (unsigned long)(0xf & m4) << 20;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrr_k_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & m3) << 20;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_a_form(unsigned long instr, unsigned char v1,unsigned char v3,unsigned char b2,unsigned int d2,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_b_form(unsigned long instr, unsigned char v1,unsigned char r3,unsigned char b2,unsigned int d2,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_c_form(unsigned long instr, unsigned char r1,unsigned char v3,unsigned char b2,unsigned int d2,unsigned char m4,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r1) << 36;
opcode += (unsigned long)(0xf & v3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m4) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrs_d_form(unsigned long instr, unsigned char r3,unsigned char b2,unsigned int d2,unsigned char v1,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & r3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & v1) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrv_form(unsigned long instr, unsigned char v1,unsigned char v2,unsigned char b2,unsigned int d2,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & v2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vrx_form(unsigned long instr, unsigned char v1,unsigned char x2,unsigned char b2,unsigned int d2,unsigned char m3,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xf & v1) << 36;
opcode += (unsigned long)(0xf & x2) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & m3) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}

LIBXSMM_API_INTERN
char libxsmm_s390x_instr_vsi_form(unsigned long instr, unsigned char i3,unsigned char b2,unsigned int d2,unsigned char v1,unsigned char rxb, unsigned char *output) {
unsigned long opcode = (0xffffffffffff & instr);
opcode += (unsigned long)(0xff & i3) << 32;
opcode += (unsigned long)(0xf & b2) << 28;
opcode += (unsigned long)(0xfff & d2) << 16;
opcode += (unsigned long)(0xf & v1) << 12;
opcode += (unsigned long)(0xf & rxb) << 8;
output[0] = 0xff & (opcode >> 40);
output[1] = 0xff & (opcode >> 32);
output[2] = 0xff & (opcode >> 24);
output[3] = 0xff & (opcode >> 16);
output[4] = 0xff & (opcode >> 8);
output[5] = 0xff & (opcode >> 0);
return 6;
}
