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
!* Hans Pabst (Intel Corp.), Alexander Heinecke (Intel Corp.)                *!
!*****************************************************************************!

PROGRAM smm
  USE :: LIBXSMM
  !$ USE omp_lib
  IMPLICIT NONE

  INTEGER, PARAMETER :: T = LIBXSMM_DOUBLE_PRECISION

  REAL(T), ALLOCATABLE, TARGET :: a(:,:,:), b(:,:,:)
  REAL(T), ALLOCATABLE, TARGET :: c(:,:), d(:,:)
  REAL(T), ALLOCATABLE, TARGET, SAVE :: tmp(:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNED_MAX :: a, b, c, tmp
  !$OMP THREADPRIVATE(tmp)
  !PROCEDURE(LIBXSMM_SMM_FUNCTION), POINTER :: xmm
  PROCEDURE(LIBXSMM_DMM_FUNCTION), POINTER :: xmm
  INTEGER :: argc, m, n, k, ld, routine, check
  INTEGER(8) :: i, s, start
  CHARACTER(32) :: argv
  TYPE(C_FUNPTR) :: f

  REAL(8) :: duration
  duration = 0

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
    READ(argv, "(I32)") i
  ELSE
    i = 2 ! 2 GByte for A and B (and C, but this currently not used by the F90 test)
  END IF
  s = LSHIFT(INT8(MAX(i, 0)), 30) / ((m * k + k * n + m * n) * T)

  ALLOCATE(c(m,n))
  ALLOCATE(a(m,k,s))
  ALLOCATE(b(k,n,s))

  ! Initialize a, b
  ! additional copies when call LIBXSMM in the F90 compiler
  !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(a, b, m, n, k, s)
  DO i = 1, s
    CALL init(42, a(:, :, i), i - 1)
    CALL init(24, b(:, :, i), i - 1)
  END DO
  c(:,:) = 0

  WRITE(*, "(A,I0,A,I0,A,I0,A,I0)") "m=", m, " n=", n, " k=", k, " size=", UBOUND(a, 3)

  ! Initialize LIBXSMM
  CALL libxsmm_init()

  ! compute reference solution
  CALL GETENV("CHECK", argv)
  READ(argv, "(I32)") check
  IF (0.NE.check) THEN
    ALLOCATE(d(m,n))
    d(:,:) = 0
    !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) SHARED(m, n, k, a, b, d)
    ALLOCATE(tmp(libxsmm_align_value(libxsmm_ld(m,n),T,LIBXSMM_ALIGNED_STORES),libxsmm_ld(n,m)))
    tmp(:,:) = 0
    !$OMP DO
    DO i = LBOUND(a, 3), UBOUND(a, 3)
      CALL libxsmm_blasmm(m, n, k, a(:,:,i), b(:,:,i), tmp)
    END DO
    !$OMP CRITICAL
    d(:,:) = d(:,:) + tmp(:UBOUND(d,1),:)
    !$OMP END CRITICAL
    ! Deallocate thread-local arrays
    DEALLOCATE(tmp)
    !$OMP END PARALLEL
  END IF

  ! warm up BLAS library
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) SHARED(m, n, k, a, b, c, duration)
  ALLOCATE(tmp(libxsmm_align_value(libxsmm_ld(m,n),T,LIBXSMM_ALIGNED_STORES),libxsmm_ld(n,m)))
  tmp(:,:) = 0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 3), UBOUND(a, 3)
    CALL libxsmm_blasmm(m, n, k, a(:,:,i), b(:,:,i), tmp)
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  !$OMP CRITICAL
  c(:,:) = c(:,:) + tmp(:UBOUND(c,1),:)
  !$OMP END CRITICAL
  ! Deallocate thread-local arrays
  DEALLOCATE(tmp)
  !$OMP END PARALLEL
  c(:,:) = 0

  WRITE(*, "(A)") "Streamed... (BLAS)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) SHARED(m, n, k, a, b, c, duration)
  ALLOCATE(tmp(libxsmm_align_value(libxsmm_ld(m,n),T,LIBXSMM_ALIGNED_STORES),libxsmm_ld(n,m)))
  tmp(:,:) = 0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 3), UBOUND(a, 3)
    CALL libxsmm_blasmm(m, n, k, a(:,:,i), b(:,:,i), tmp)
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  !$OMP CRITICAL
  c(:,:) = c(:,:) + tmp(:UBOUND(c,1),:)
  !$OMP END CRITICAL
  ! Deallocate thread-local arrays
  DEALLOCATE(tmp)
  !$OMP END PARALLEL
  CALL performance(duration, m, n, k, s)
  IF (0.NE.check) THEN
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", MAXVAL((c(:,:) - d(:,:)) * (c(:,:) - d(:,:)))
  END IF
  c(:,:) = 0

  WRITE(*, "(A)") "Streamed... (inlined)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) SHARED(m, n, k, a, b, c, duration)
  ALLOCATE(tmp(libxsmm_align_value(libxsmm_ld(m,n),T,LIBXSMM_ALIGNED_STORES),libxsmm_ld(n,m)))
  tmp(:,:) = 0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 3), UBOUND(a, 3)
    CALL libxsmm_imm(m, n, k, a(:,:,i), b(:,:,i), tmp)
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  !$OMP CRITICAL
  c(:,:) = c(:,:) + tmp(:UBOUND(c,1),:)
  !$OMP END CRITICAL
  ! Deallocate thread-local arrays
  DEALLOCATE(tmp)
  !$OMP END PARALLEL
  CALL performance(duration, m, n, k, s)
  IF (0.NE.check) THEN
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", MAXVAL((c(:,:) - d(:,:)) * (c(:,:) - d(:,:)))
  END IF
  c(:,:) = 0

  WRITE(*, "(A)") "Streamed... (auto-dispatched)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) SHARED(m, n, k, a, b, c, duration)
  ALLOCATE(tmp(libxsmm_align_value(libxsmm_ld(m,n),T,LIBXSMM_ALIGNED_STORES),libxsmm_ld(n,m)))
  tmp(:,:) = 0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 3), UBOUND(a, 3)
    CALL libxsmm_mm(m, n, k, a(:,:,i), b(:,:,i), tmp)
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  !$OMP CRITICAL
  c(:,:) = c(:,:) + tmp(:UBOUND(c,1),:)
  !$OMP END CRITICAL
  ! Deallocate thread-local arrays
  DEALLOCATE(tmp)
  !$OMP END PARALLEL
  CALL performance(duration, m, n, k, s)
  IF (0.NE.check) THEN
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", MAXVAL((c(:,:) - d(:,:)) * (c(:,:) - d(:,:)))
  END IF
  c(:,:) = 0

  f = libxsmm_dispatch(m, n, k, T)
  IF (C_ASSOCIATED(f)) THEN
    CALL C_F_PROCPOINTER(f, xmm)
    WRITE(*, "(A)") "Streamed... (specialized)"
    !$OMP PARALLEL PRIVATE(i, start) !DEFAULT(NONE) SHARED(m, n, a, b, c, duration, xmm)
    ALLOCATE(tmp(libxsmm_align_value(libxsmm_ld(m,n),T,LIBXSMM_ALIGNED_STORES),libxsmm_ld(n,m)))
    tmp(:,:) = 0
    !$OMP MASTER
    start = libxsmm_timer_tick()
    !$OMP END MASTER
    !$OMP DO
    DO i = LBOUND(a, 3), UBOUND(a, 3)
      CALL xmm(a(:,:,i), b(:,:,i), tmp)
    END DO
    !$OMP MASTER
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
    !$OMP END MASTER
    !$OMP CRITICAL
    c(:,:) = c(:,:) + tmp(:UBOUND(c,1),:)
    !$OMP END CRITICAL
    ! Deallocate thread-local arrays
    DEALLOCATE(tmp)
    !$OMP END PARALLEL
    CALL performance(duration, m, n, k, s)
    IF (0.NE.check) THEN
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", MAXVAL((c(:,:) - d(:,:)) * (c(:,:) - d(:,:)))
    END IF
    c(:,:) = 0
  END IF

  ! Deallocate global arrays
  DEALLOCATE(a)
  DEALLOCATE(b)
  DEALLOCATE(c)
  IF (0.NE.check) THEN
    DEALLOCATE(d)
  END IF

CONTAINS
  PURE SUBROUTINE init(seed, matrix, n)
    INTEGER, INTENT(IN) :: seed
    REAL(T), INTENT(OUT) :: matrix(:,:)
    INTEGER(8), INTENT(IN), OPTIONAL :: n
    INTEGER(8) :: minval, addval, maxval
    INTEGER :: ld, i, j
    REAL(8) :: value, norm
    ld = UBOUND(matrix, 1) - LBOUND(matrix, 1) + 1
    minval = MERGE(n, 0_8, PRESENT(n)) + seed
    addval = (UBOUND(matrix, 1) - LBOUND(matrix, 1)) * ld + (UBOUND(matrix, 2) - LBOUND(matrix, 2))
    maxval = MAX(ABS(minval), addval)
    norm = MERGE(1D0 / maxval, 1D0, 0.NE.maxval)
    DO j = LBOUND(matrix, 2), UBOUND(matrix, 2)
      DO i = LBOUND(matrix, 1), LBOUND(matrix, 1) + UBOUND(matrix, 1) - 1
        value = (i - LBOUND(matrix, 1)) * ld + (j - LBOUND(matrix, 2)) + minval
        matrix(i,j) = norm * (value - 0.5D0 * addval)
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
      fmt = "(16F20.0)"
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

  SUBROUTINE performance(duration, m, n, k, s)
    REAL(T), INTENT(IN) :: duration
    INTEGER, INTENT(IN) :: m, n, k
    INTEGER(8), INTENT(IN) :: s
    IF (0.LT.duration) THEN
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:", &
        (2D0 * s * m * n * k * 1D-9 / duration), " GFLOPS/s"
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ", &
        (s * (m * k + k * n) * T / (duration * LSHIFT(1_8, 30))), " GB/s"
    ENDIF
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ", 1D3 * duration, " ms"
  END SUBROUTINE
END PROGRAM
