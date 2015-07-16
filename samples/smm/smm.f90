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

  INTEGER, PARAMETER :: realtype  = LIBXSMM_DOUBLE_PRECISION
  INTEGER, PARAMETER :: m = 7, n = 11, k = 23
  REAL(realtype), ALLOCATABLE :: a(:,:), b(:,:), c(:,:), d(:,:)
  INTEGER :: lda, ldb, ldc
  REAL(realtype) :: diff

  lda = m
  ldb = k
  ldc = libxsmm_ldc(m, n, realtype)

  ALLOCATE(a(lda,k))
  ALLOCATE(b(ldb,n))
  ALLOCATE(c(ldc,n))
  ALLOCATE(d(ldc,n))

  ! Initialize matrices
  CALL init(a, 42)
  CALL init(b, 24)

  ! Calculate reference based on BLAS
  d(:,:) = 0
  CALL libxsmm_blasmm(m, n, k, a, b, d)

  c(:,:) = 0
  CALL libxsmm_imm(m, n, k, a, b, c)
  WRITE(*,*) "diff = ", MAXVAL(((c(:,:) - d(:,:)) * (c(:,:) - d(:,:)))), "(optimized)"

  c(:,:) = 0
  CALL libxsmm_mm(m, n, k, a, b, c)
  WRITE(*,*) "diff = ", MAXVAL(((c(:,:) - d(:,:)) * (c(:,:) - d(:,:)))), "(dispatched)"

  DEALLOCATE(a)
  DEALLOCATE(b)
  DEALLOCATE(c)
  DEALLOCATE(d)

CONTAINS
  SUBROUTINE init(matrix, seed)
    REAL(realtype), INTENT(OUT) :: matrix(:,:)
    INTEGER, INTENT(IN) :: seed
    INTEGER :: i, j
    DO j = LBOUND(matrix, 2), UBOUND(matrix, 2)
      DO i = LBOUND(matrix, 1), UBOUND(matrix, 1)
        matrix(i,j) = i * SIZE(matrix, 1) + j + SIZE(matrix, 2) + seed
      END DO
    END DO
  END SUBROUTINE
END PROGRAM
