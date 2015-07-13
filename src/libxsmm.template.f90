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
  !USE, INTRINSIC :: ISO_C_BINDING
  IMPLICIT NONE

  ! Kind of types used to parameterize the implementation.
  INTEGER, PARAMETER :: LIBXSMM_SINGLE_PRECISION  = KIND(1.0)
  INTEGER, PARAMETER :: LIBXSMM_DOUBLE_PRECISION  = KIND(1D0)
  INTEGER, PARAMETER :: LIBXSMM_INTEGER_TYPE      = KIND(1)

  ! Parameters the library was built for.
  INTEGER, PARAMETER :: LIBXSMM_ALIGNMENT       = $ALIGNMENT
  INTEGER, PARAMETER :: LIBXSMM_ALIGNED_STORES  = $ALIGNED_STORES
  INTEGER, PARAMETER :: LIBXSMM_ALIGNED_LOADS   = $ALIGNED_LOADS
  INTEGER, PARAMETER :: LIBXSMM_ALIGNED_MAX     = $ALIGNED_MAX
  INTEGER, PARAMETER :: LIBXSMM_ROW_MAJOR       = $ROW_MAJOR
  INTEGER, PARAMETER :: LIBXSMM_COL_MAJOR       = $COL_MAJOR
  INTEGER, PARAMETER :: LIBXSMM_MAX_MNK         = $MAX_MNK
  INTEGER, PARAMETER :: LIBXSMM_MAX_M           = $MAX_M
  INTEGER, PARAMETER :: LIBXSMM_MAX_N           = $MAX_N
  INTEGER, PARAMETER :: LIBXSMM_MAX_K           = $MAX_K
  INTEGER, PARAMETER :: LIBXSMM_AVG_M           = $AVG_M
  INTEGER, PARAMETER :: LIBXSMM_AVG_N           = $AVG_N
  INTEGER, PARAMETER :: LIBXSMM_AVG_K           = $AVG_K

  INTERFACE LIBXSMM_BLASMM
    MODULE PROCEDURE LIBXSMM_SBLASMM, LIBXSMM_DBLASMM
  END INTERFACE

  INTERFACE LIBXSMM_IMM
    MODULE PROCEDURE LIBXSMM_SIMM, LIBXSMM_DIMM
  END INTERFACE

  INTERFACE LIBXSMM_MM
    MODULE PROCEDURE LIBXSMM_SMM, LIBXSMM_DMM
  END INTERFACE

  ABSTRACT INTERFACE
    ! Type of a function generated for a specific M, N, and K.
    PURE SUBROUTINE LIBXSMM_SMM_FUNCTION(a, b, c)
      USE, INTRINSIC :: ISO_C_BINDING
      REAL(C_FLOAT), INTENT(IN) :: a, b
      REAL(C_FLOAT), INTENT(INOUT) :: c
    END SUBROUTINE

    ! Type of a function generated for a specific M, N, and K.
    PURE SUBROUTINE LIBXSMM_DMM_FUNCTION(a, b, c)
      USE, INTRINSIC :: ISO_C_BINDING
      REAL(C_DOUBLE), INTENT(IN) :: a, b
      REAL(C_DOUBLE), INTENT(INOUT) :: c
    END SUBROUTINE
  END INTERFACE$MNK_INTERFACE_LIST

CONTAINS
  ! Non-dispatched matrix-matrix multiplication using BLAS; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXSMM_SBLASMM
  !DIR$ ATTRIBUTES INLINE :: LIBXSMM_SBLASMM
  PURE SUBROUTINE LIBXSMM_SBLASMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXSMM_SINGLE_PRECISION), INTENT(IN) :: a, b
    REAL(LIBXSMM_SINGLE_PRECISION), INTENT(INOUT) :: c
    !LIBXSMM_BLASMM(LIBXSMM_SINGLE_PRECISION, m, n, k, a, b, c)
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using BLAS; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXSMM_DBLASMM
  !DIR$ ATTRIBUTES INLINE :: LIBXSMM_DBLASMM
  PURE SUBROUTINE LIBXSMM_DBLASMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(IN) :: a, b
    REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(INOUT) :: c
    !LIBXSMM_BLASMM(LIBXSMM_DOUBLE_PRECISION, m, n, k, a, b, c)
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using inline code; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXSMM_SIMM
  !DIR$ ATTRIBUTES INLINE :: LIBXSMM_SIMM
  PURE SUBROUTINE LIBXSMM_SIMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXSMM_SINGLE_PRECISION), INTENT(IN) :: a, b
    REAL(LIBXSMM_SINGLE_PRECISION), INTENT(INOUT) :: c
    !LIBXSMM_IMM(LIBXSMM_SINGLE_PRECISION, LIBXSMM_INTEGER_TYPE, m, n, k, a, b, c)
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using inline code; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXSMM_DIMM
  !DIR$ ATTRIBUTES INLINE :: LIBXSMM_DIMM
  PURE SUBROUTINE LIBXSMM_DIMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(IN) :: a, b
    REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(INOUT) :: c
    !LIBXSMM_IMM(LIBXSMM_DOUBLE_PRECISION, LIBXSMM_INTEGER_TYPE, m, n, k, a, b, c)
  END SUBROUTINE

  ! Query the pointer of a generated function; zero if it does not exist.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXSMM_SMM_DISPATCH
  !DIR$ ATTRIBUTES INLINE :: LIBXSMM_SMM_DISPATCH
  FUNCTION LIBXSMM_SMM_DISPATCH(m, n, k) RESULT(f)
    PROCEDURE(LIBXSMM_SMM_FUNCTION), POINTER :: f
    INTEGER, INTENT(IN) :: m, n, k
    INTERFACE
      !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXSMM_SMM_DISPATCH_AUX
      TYPE(C_FUNPTR) PURE FUNCTION LIBXSMM_SMM_DISPATCH_AUX(m, n, k) BIND(C, NAME="libxsmm_smm_dispatch")
        USE, INTRINSIC :: ISO_C_BINDING
        INTEGER(C_INT), INTENT(IN) :: m, n, k
      END FUNCTION
    END INTERFACE
    CALL C_F_PROCPOINTER(LIBXSMM_SMM_DISPATCH_AUX(m, n, k), f)
  END FUNCTION

  ! Query the pointer of a generated function; zero if it does not exist.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXSMM_DMM_DISPATCH
  !DIR$ ATTRIBUTES INLINE :: LIBXSMM_DMM_DISPATCH
  FUNCTION LIBXSMM_DMM_DISPATCH(m, n, k) RESULT(f)
    PROCEDURE(LIBXSMM_DMM_FUNCTION), POINTER :: f
    INTEGER, INTENT(IN) :: m, n, k
    INTERFACE
      !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXSMM_DMM_DISPATCH_AUX
      TYPE(C_FUNPTR) PURE FUNCTION LIBXSMM_DMM_DISPATCH_AUX(m, n, k) BIND(C, NAME="libxsmm_dmm_dispatch")
        USE, INTRINSIC :: ISO_C_BINDING
        INTEGER(C_INT), INTENT(IN) :: m, n, k
      END FUNCTION
    END INTERFACE
    CALL C_F_PROCPOINTER(LIBXSMM_DMM_DISPATCH_AUX(m, n, k), f)
  END FUNCTION

  ! Dispatched matrix-matrix multiplication; single-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXSMM_SMM
  !DIR$ ATTRIBUTES INLINE :: LIBXSMM_SMM
  SUBROUTINE LIBXSMM_SMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXSMM_SINGLE_PRECISION), INTENT(IN) :: a, b
    REAL(LIBXSMM_SINGLE_PRECISION), INTENT(INOUT) :: c
    PROCEDURE(LIBXSMM_SMM_FUNCTION), POINTER :: f
    IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
      f => LIBXSMM_SMM_DISPATCH(m, n, k)
      IF (ASSOCIATED(f)) THEN
        CALL f(a, b, c)
      ELSE
        CALL LIBXSMM_SIMM(M, N, K, A, B, C)
      ENDIF
    ELSE
      CALL LIBXSMM_SBLASMM(M, N, K, A, B, C)
    ENDIF
  END SUBROUTINE

  ! Dispatched matrix-matrix multiplication; double-precision.
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: LIBXSMM_DMM
  !DIR$ ATTRIBUTES INLINE :: LIBXSMM_DMM
  SUBROUTINE LIBXSMM_DMM(m, n, k, a, b, c)
    INTEGER, INTENT(IN) :: m, n, k
    REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(IN) :: a, b
    REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(INOUT) :: c
    PROCEDURE(LIBXSMM_DMM_FUNCTION), POINTER :: f
    IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
      f => LIBXSMM_DMM_DISPATCH(m, n, k)
      IF (ASSOCIATED(f)) THEN
        CALL f(a, b, c)
      ELSE
        CALL LIBXSMM_DIMM(M, N, K, A, B, C)
      ENDIF
    ELSE
      CALL LIBXSMM_DBLASMM(M, N, K, A, B, C)
    ENDIF
  END SUBROUTINE
END MODULE
