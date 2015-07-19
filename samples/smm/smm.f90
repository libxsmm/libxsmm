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
  INTEGER :: argc, m, n, k, routine
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

  ALLOCATE(a(libxsmm_ld(m, k),libxsmm_ld(k, m)))
  ALLOCATE(b(libxsmm_ld(k, n),libxsmm_ld(n, k)))
  ALLOCATE(c(libxsmm_ldc(m, n, T),libxsmm_ld(n, m)))
  ALLOCATE(d(libxsmm_ldc(m, n, T),libxsmm_ld(n, m)))

  ! Initialize matrices
  CALL init(a, 42)
  CALL init(b, 24)

  ! Calculate reference based on BLAS
  d(:,:) = 0
  CALL libxsmm_blasmm(m, n, k, a, b, d)

  c(:,:) = 0
  IF (0.LT.routine) THEN
    CALL libxsmm_mm(m, n, k, a, b, c)
  ELSE
    f = MERGE(libxsmm_mm_dispatch(m, n, k, T), C_NULL_FUNPTR, 0.EQ.routine)
    IF (C_ASSOCIATED(f)) THEN
      CALL C_F_PROCPOINTER(f, xmm)
      CALL xmm(C_LOC(a), C_LOC(b), C_LOC(c))
    ELSE
      CALL libxsmm_imm(m, n, k, a, b, c)
    ENDIF
  END IF

  WRITE(*,*) "diff = ", MAXVAL(((c(:,:) - d(:,:)) * (c(:,:) - d(:,:))))

  DEALLOCATE(a)
  DEALLOCATE(b)
  DEALLOCATE(c)
  DEALLOCATE(d)

CONTAINS
  PURE SUBROUTINE init(matrix, seed)
    REAL(T), INTENT(OUT) :: matrix(:,:)
    INTEGER, INTENT(IN) :: seed
    INTEGER :: i, j
    DO j = LBOUND(matrix, 2), UBOUND(matrix, 2)
      DO i = LBOUND(matrix, 1), UBOUND(matrix, 1)
        matrix(i,j) = SIZE(matrix, 2) * i + j + seed - 1
      END DO
    END DO
  END SUBROUTINE

  SUBROUTINE disp(matrix, format)
    REAL(T), INTENT(IN) :: matrix(:,:)
    CHARACTER(*), OPTIONAL, INTENT(IN) :: format
    CHARACTER(32) :: fmt
    INTEGER :: i
    IF (.NOT.PRESENT(format)) THEN
      fmt = "(16F8.0)"
    ELSE
      WRITE(fmt, "('(16',A,')')") format
    ENDIF
    DO i = LBOUND(matrix, 1), UBOUND(matrix, 1)
      WRITE(*, fmt) matrix(i,:)
    END DO
  END SUBROUTINE
END PROGRAM
