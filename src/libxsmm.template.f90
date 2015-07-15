!*****************************************************************************!
!* Copyright (c) 2013-2015, Intel Corporation                                *!
!* All rights reserved.                                                      *!
!*                                                                           *!
!* Redistribution and use in source and binary forms, with or without        *!
!* modification, are permitted provided that the following conditions        *!
!* are met:                                                                  *!
!* 1. Redistributions of source code must retain the above copyright         *!
!*    notice, this list of conditions and the following disclaimer.          *!
!* 2. Redistributions in binary form must reproduce the above copyright      *!
!*    notice, this list of conditions and the following disclaimer in the    *!
!*    documentation and/or other materials provided with the distribution.   *!
!* 3. Neither the name of the copyright holder nor the names of its          *!
!*    contributors may be used to endorse or promote products derived        *!
!*    from this software without specific prior written permission.          *!
!*                                                                           *!
!* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       *!
!* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         *!
!* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     *!
!* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      *!
!* HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    *!
!* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  *!
!* TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    *!
!* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    *!
!* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      *!
!* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        *!
!* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              *!
!*****************************************************************************!
!* Hans Pabst (Intel Corp.)                                                  *!
!*****************************************************************************!

MODULE LIBXSMM
  USE, INTRINSIC :: ISO_C_BINDING
  IMPLICIT NONE

  ! Kind of types used to parameterize the implementation.
  INTEGER, PARAMETER :: LIBXSMM_SINGLE_PRECISION  = KIND(1.0)
  INTEGER, PARAMETER :: LIBXSMM_DOUBLE_PRECISION  = KIND(1D0)
  INTEGER, PARAMETER :: LIBXSMM_INTEGER_TYPE      = KIND(1)

  ! Parameters the library was built for.
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_ALIGNMENT       = $ALIGNMENT
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_ALIGNED_STORES  = $ALIGNED_STORES
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_ALIGNED_LOADS   = $ALIGNED_LOADS
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_ALIGNED_MAX     = $ALIGNED_MAX
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_ROW_MAJOR       = $ROW_MAJOR
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_COL_MAJOR       = $COL_MAJOR
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_MAX_MNK         = $MAX_MNK
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_MAX_M           = $MAX_M
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_MAX_N           = $MAX_N
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_MAX_K           = $MAX_K
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_AVG_M           = $AVG_M
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_AVG_N           = $AVG_N
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_AVG_K           = $AVG_K

  ! Overloaded BLAS routines (single/double precision)
  INTERFACE libxsmm_blasmm
    MODULE PROCEDURE libxsmm_sblasmm, libxsmm_dblasmm
  END INTERFACE

  ! Overloaded inlinable routines (single/double precision)
  INTERFACE libxsmm_imm
    MODULE PROCEDURE libxsmm_simm, libxsmm_dimm
  END INTERFACE

  ! Overloaded auto-dispatch routines (single/double precision)
  INTERFACE libxsmm_mm
    MODULE PROCEDURE libxsmm_smm, libxsmm_dmm
  END INTERFACE

  ABSTRACT INTERFACE
    ! Type of a function generated for a specific M, N, and K
    PURE SUBROUTINE LIBXSMM_XMM_FUNCTION(a, b, c)
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE, INTENT(IN) :: a, b, c
    END SUBROUTINE
  END INTERFACE$MNK_INTERFACE_LIST

CONTAINS
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_up
  !DIR$ ATTRIBUTES INLINE :: libxsmm_up
  PURE FUNCTION libxsmm_up(n, up) RESULT(nup)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: n, up
    INTEGER(LIBXSMM_INTEGER_TYPE) :: nup
    nup = ((n + up - 1) / up) * up
  END FUNCTION

  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_align_value
  !DIR$ ATTRIBUTES INLINE :: libxsmm_align_value
  PURE FUNCTION libxsmm_align_value(n, typesize, alignment) RESULT(na)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: n, typesize, alignment
    INTEGER(LIBXSMM_INTEGER_TYPE) :: na
    na = libxsmm_up(n * typesize, alignment) / typesize
  END FUNCTION

  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ldc
  !DIR$ ATTRIBUTES INLINE :: libxsmm_ldc
  PURE FUNCTION libxsmm_ldc(m, n, typesize) RESULT(ldc)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, typesize
    INTEGER(LIBXSMM_INTEGER_TYPE) :: ldc
    ldc = libxsmm_align_value(MERGE(m, n, 0.NE.LIBXSMM_COL_MAJOR), typesize, LIBXSMM_ALIGNED_STORES)
  END FUNCTION

  ! Non-dispatched matrix-matrix multiplication using BLAS; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sblasmm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_sblasmm
  SUBROUTINE libxsmm_sblasmm(m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(LIBXSMM_SINGLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXSMM_SINGLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    REAL(LIBXSMM_SINGLE_PRECISION), PARAMETER :: alpha = 1, beta = 1
    CALL sgemm('N', 'N', MERGE(m, n, 0.NE.LIBXSMM_COL_MAJOR), MERGE(n, m, 0.NE.LIBXSMM_COL_MAJOR), k, alpha, &
      MERGE(a, b, 0.NE.LIBXSMM_COL_MAJOR), MERGE(m, n, 0.NE.LIBXSMM_COL_MAJOR), &
      MERGE(b, a, 0.NE.LIBXSMM_COL_MAJOR), k, &
      beta, c, libxsmm_ldc(m, n, LIBXSMM_SINGLE_PRECISION))
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using BLAS; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dblasmm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_dblasmm
  SUBROUTINE libxsmm_dblasmm(m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    REAL(LIBXSMM_DOUBLE_PRECISION), PARAMETER :: alpha = 1, beta = 1
    CALL dgemm('N', 'N', MERGE(m, n, 0.NE.LIBXSMM_COL_MAJOR), MERGE(n, m, 0.NE.LIBXSMM_COL_MAJOR), k, alpha, &
      MERGE(a, b, 0.NE.LIBXSMM_COL_MAJOR), MERGE(m, n, 0.NE.LIBXSMM_COL_MAJOR), &
      MERGE(b, a, 0.NE.LIBXSMM_COL_MAJOR), k, &
      beta, c, libxsmm_ldc(m, n, LIBXSMM_DOUBLE_PRECISION))
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using inline code; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_simm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_simm
  PURE SUBROUTINE libxsmm_simm(m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(LIBXSMM_SINGLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXSMM_SINGLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    !LIBXSMM_IMM(LIBXSMM_SINGLE_PRECISION, LIBXSMM_INTEGER_TYPE, m, n, k, a, b, c)
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using inline code; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dimm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_dimm
  PURE SUBROUTINE libxsmm_dimm(m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(INOUT) :: c($SHAPE_C)
    !LIBXSMM_IMM(LIBXSMM_DOUBLE_PRECISION, LIBXSMM_INTEGER_TYPE, m, n, k, a, b, c)
  END SUBROUTINE

  ! Query the pointer of a generated function; zero if it does not exist.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm_dispatch
  !DIR$ ATTRIBUTES INLINE :: libxsmm_smm_dispatch
  FUNCTION libxsmm_smm_dispatch(m, n, k) RESULT(f)
    PROCEDURE(LIBXSMM_XMM_FUNCTION), POINTER :: f
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    INTERFACE
      !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm_dispatch_aux
      TYPE(C_FUNPTR) PURE FUNCTION libxsmm_smm_dispatch_aux(m, n, k) BIND(C, NAME="libxsmm_smm_dispatch")
        IMPORT :: C_FUNPTR, C_INT
        INTEGER(C_INT), VALUE, INTENT(IN) :: m, n, k
      END FUNCTION
    END INTERFACE
    CALL C_F_PROCPOINTER(libxsmm_smm_dispatch_aux(m, n, k), f)
  END FUNCTION

  ! Query the pointer of a generated function; zero if it does not exist.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmm_dispatch
  !DIR$ ATTRIBUTES INLINE :: libxsmm_dmm_dispatch
  FUNCTION libxsmm_dmm_dispatch(m, n, k) RESULT(f)
    PROCEDURE(LIBXSMM_XMM_FUNCTION), POINTER :: f
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    INTERFACE
      !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmm_dispatch_aux
      TYPE(C_FUNPTR) PURE FUNCTION libxsmm_dmm_dispatch_aux(m, n, k) BIND(C, NAME="libxsmm_dmm_dispatch")
        IMPORT :: C_FUNPTR, C_INT
        INTEGER(C_INT), VALUE, INTENT(IN) :: m, n, k
      END FUNCTION
    END INTERFACE
    CALL C_F_PROCPOINTER(libxsmm_dmm_dispatch_aux(m, n, k), f)
  END FUNCTION

  ! Dispatched matrix-matrix multiplication; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_smm
  SUBROUTINE libxsmm_smm(m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(LIBXSMM_SINGLE_PRECISION), TARGET, INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXSMM_SINGLE_PRECISION), TARGET, INTENT(INOUT) :: c($SHAPE_C)
    PROCEDURE(LIBXSMM_XMM_FUNCTION), POINTER :: xmm
    IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
      xmm => libxsmm_smm_dispatch(m, n, k)
      IF (ASSOCIATED(xmm)) THEN
        CALL xmm(C_LOC(a), C_LOC(b), C_LOC(c))
      ELSE
        CALL libxsmm_simm(m, n, k, a, b, c)
      ENDIF
    ELSE
      CALL libxsmm_sblasmm(m, n, k, a, b, c)
    ENDIF
  END SUBROUTINE

  ! Dispatched matrix-matrix multiplication; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_dmm
  SUBROUTINE libxsmm_dmm(m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(LIBXSMM_DOUBLE_PRECISION), TARGET, INTENT(IN) :: a($SHAPE_A), b($SHAPE_B)
    REAL(LIBXSMM_DOUBLE_PRECISION), TARGET, INTENT(INOUT) :: c($SHAPE_C)
    PROCEDURE(LIBXSMM_XMM_FUNCTION), POINTER :: xmm
    IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
      xmm => libxsmm_dmm_dispatch(m, n, k)
      IF (ASSOCIATED(xmm)) THEN
        CALL xmm(C_LOC(a), C_LOC(b), C_LOC(c))
      ELSE
        CALL libxsmm_dimm(m, n, k, a, b, c)
      ENDIF
    ELSE
      CALL libxsmm_dblasmm(m, n, k, a, b, c)
    ENDIF
  END SUBROUTINE
END MODULE
