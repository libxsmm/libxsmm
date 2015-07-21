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

PROGRAM smm
  USE :: LIBXSMM
  IMPLICIT NONE

  INTEGER, PARAMETER :: T = LIBXSMM_DOUBLE_PRECISION
  REAL(T), ALLOCATABLE, TARGET :: a(:,:), b(:,:), c(:,:), d(:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNED_MAX :: a, b, c, d
  PROCEDURE(LIBXSMM_XMM_FUNCTION), POINTER :: xmm
  INTEGER :: argc, m, n, k, ld, routine
  CHARACTER(32) :: argv
  TYPE(C_FUNPTR) :: f

  argc = IARGC()
  IF (1 <= argc) THEN
    CALL GETARG(1, argv)
    READ(argv, "(I32)") m
  ELSE
    m = 23
  END IF
  IF (2 <= argc) THEN
    CALL GETARG(2, argv)
    READ(argv, "(I32)") n
  ELSE
    n = m
  END IF
  IF (3 <= argc) THEN
    CALL GETARG(3, argv)
    READ(argv, "(I32)") k
  ELSE
    k = m
  END IF
  IF (4 <= argc) THEN
    CALL GETARG(4, argv)
    READ(argv, "(I32)") routine
  ELSE
    routine = -1
  END IF

  ALLOCATE(a(m,k))
  ALLOCATE(b(k,n))
  ALLOCATE(c(libxsmm_align_value(m,T,LIBXSMM_ALIGNED_STORES),n))
  ALLOCATE(d(libxsmm_align_value(m,T,LIBXSMM_ALIGNED_STORES),n))

  ! Initialize matrices
  CALL init(42, a)
  CALL init(24, b)

  ! Calculate reference based on BLAS
  d(:,:) = 0
  CALL libxsmm_blasmm(m, n, k, a, b, d)

  c(:,:) = 0
  IF (0.GT.routine) THEN
    WRITE(*,*) "auto-dispatched"
    CALL libxsmm_mm(m, n, k, a, b, c)
  ELSE
    f = MERGE(libxsmm_mm_dispatch(m, n, k, T), C_NULL_FUNPTR, 0.EQ.routine)
    IF (C_ASSOCIATED(f)) THEN
      WRITE(*,*) "specialized"
      CALL C_F_PROCPOINTER(f, xmm)
      CALL xmm(C_LOC(a), C_LOC(b), C_LOC(c))
    ELSE
      IF (0.EQ.routine) THEN
        WRITE(*,*) "optimized (no specialized routine found)"
      ELSE
        WRITE(*,*) "optimized"
      ENDIF
      CALL libxsmm_imm(m, n, k, a, b, c)
    ENDIF
  END IF

  ld = LBOUND(c, 1) + m - 1
  WRITE(*,*) "diff = ", MAXVAL(((c(:ld,:) - d(:ld,:)) * (c(:ld,:) - d(:ld,:))))

  DEALLOCATE(a)
  DEALLOCATE(b)
  DEALLOCATE(c)
  DEALLOCATE(d)

CONTAINS
  PURE SUBROUTINE init(seed, matrix, ld, n)
    INTEGER, INTENT(IN) :: seed
    REAL(T), INTENT(OUT) :: matrix(:,:)
    INTEGER, INTENT(IN), OPTIONAL :: ld, n
    INTEGER :: shift, i0, i1, i, j
    i0 = LBOUND(matrix, 1)
    i1 = MIN(MERGE(i0 + ld - 1, UBOUND(matrix, 1), PRESENT(ld)), UBOUND(matrix, 1))
    shift = seed + MERGE(n, 0, PRESENT(n)) - LBOUND(matrix, 1)
    DO j = LBOUND(matrix, 2), UBOUND(matrix, 2)
      DO i = i0, i1
        matrix(i,j) = (j - LBOUND(matrix, 2)) * SIZE(matrix, 1) + i + shift
      END DO
    END DO
  END SUBROUTINE

  SUBROUTINE disp(matrix, ld, format)
    REAL(T), INTENT(IN) :: matrix(:,:)
    INTEGER, INTENT(IN), OPTIONAL :: ld
    CHARACTER(*), INTENT(IN), OPTIONAL :: format
    CHARACTER(32) :: fmt
    INTEGER :: i0, i1, i, j
    IF (.NOT.PRESENT(format)) THEN
      fmt = "(16F8.0)"
    ELSE
      WRITE(fmt, "('(16',A,')')") format
    ENDIF
    i0 = LBOUND(matrix, 1)
    i1 = MIN(MERGE(i0 + ld - 1, UBOUND(matrix, 1), PRESENT(ld)), UBOUND(matrix, 1))
    DO i = i0, i1
      DO j = LBOUND(matrix, 2), UBOUND(matrix, 2)
        WRITE(*, fmt, advance='NO') matrix(i,j)
      END DO
      WRITE(*, *)
    END DO
  END SUBROUTINE
END PROGRAM
