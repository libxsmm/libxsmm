!*****************************************************************************!
!* Copyright (c) 2015-2016, Intel Corporation                                *!
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
!* Hans Pabst (Intel Corp.), Alexander Heinecke (Intel Corp.), and           *!
!* Maxwell Hutchinson (University of Chicago)                                *!
!*****************************************************************************!


PROGRAM stpm
  USE :: LIBXSMM
  USE :: STREAM_UPDATE_KERNELS

  !$ USE omp_lib
  IMPLICIT NONE

  INTEGER, PARAMETER :: T = KIND(0D0)
  REAL(T), PARAMETER :: alpha = 1, beta = 0

  REAL(T), ALLOCATABLE, DIMENSION(:,:,:,:), TARGET :: a, c, d
  REAL(T), ALLOCATABLE, TARGET :: dx(:,:), dy(:,:), dz(:,:)
  REAL(T), ALLOCATABLE, TARGET, SAVE :: tm1(:,:,:), tm2(:,:,:), tm3(:,:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNMENT :: a, c, d
  !$OMP THREADPRIVATE(tm1, tm2, tm3)
  TYPE(LIBXSMM_DMMFUNCTION) :: xmm1, xmm2, xmm3
  DOUBLE PRECISION :: duration, max_diff
  INTEGER :: argc, m, n, k, routine, check, mm, nn, kk
  INTEGER(8) :: i, j, ix, iy, iz, r, s, size0, size1, size, repetitions, start
  CHARACTER(32) :: argv

  argc = COMMAND_ARGUMENT_COUNT()
  IF (1 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(1, argv)
    READ(argv, "(I32)") m
  ELSE
    m = 8
  END IF
  IF (2 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(2, argv)
    READ(argv, "(I32)") n
  ELSE
    n = m
  END IF
  IF (3 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(3, argv)
    READ(argv, "(I32)") k
  ELSE
    k = m
  END IF
  mm = 0
  IF (4 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(4, argv)
    READ(argv, "(I32)") mm
  END IF
  mm = MERGE(10, mm, 0.EQ.mm)
  nn = 0
  IF (5 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(5, argv)
    READ(argv, "(I32)") nn
  END IF
  nn = MERGE(mm, nn, 0.EQ.nn)
  kk = 0
  IF (6 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(6, argv)
    READ(argv, "(I32)") kk
  END IF
  kk = MERGE(mm, kk, 0.EQ.kk)
  IF (7 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(7, argv)
    READ(argv, "(I32)") size1
  ELSE
    size1 = 0
  END IF
  IF (8 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(8, argv)
    READ(argv, "(I32)") size
  ELSE
    size = 0 ! 1 repetition by default
  END IF

  ! Initialize LIBXSMM
  CALL libxsmm_init()

  ! workload is about 2 GByte in memory by default
  size0 = ((m * n * k) + (nn * mm * kk)) * T ! size of a single stream element in Byte
  size1 = MERGE(2048_8, MERGE(size1, ISHFT(ABS(size0 * size1) + ISHFT(1, 20) - 1, -20), 0.LE.size1), 0.EQ.size1)
  size = ISHFT(MERGE(MAX(size, size1), ISHFT(ABS(size) * size0 + ISHFT(1, 20) - 1, -20), 0.LE.size), 20) / size0
  s = ISHFT(size1, 20) / size0; repetitions = size / s; duration = 0; max_diff = 0

  ALLOCATE(a(m,n,k,s))
  ALLOCATE(c(mm,nn,kk,s))
  ALLOCATE(dx(mm,m), dy(n,nn), dz(k,kk))

  ! Initialize
  !$OMP PARALLEL DO PRIVATE(i, ix, iy, iz) DEFAULT(NONE) SHARED(a, m, mm, n, nn, k, kk, s)
  DO i = 1, s
    DO ix = 1, m
      DO iy = 1, n
        DO iz = 1, k
          a(ix,iy,iz,i) = ix + iy*m + iz*m*n
        END DO
      END DO
    END DO
  END DO
  !$OMP PARALLEL DO PRIVATE(i, ix, iy, iz) DEFAULT(NONE) SHARED(c, m, mm, n, nn, k, kk, s)
  DO i = 1, s
    DO ix = 1, mm
      DO iy = 1, nn
        DO iz = 1, kk
          c(ix,iy,iz,i) = 0.0
        END DO
      END DO
    END DO
  END DO
  dx = 1.
  dy = 1.
  dz = 1.

  WRITE(*, "(6(A,I0),A,I0,A,I0,A,I0)") "m=", m, " n=", n, " k=", k, " mm=", mm, " nn=", nn, " kk=", kk, &
    " elements=", UBOUND(a, 4), " size=", size1, "MB repetitions=", repetitions

  CALL GETENV("CHECK", argv)
  READ(argv, "(I32)") check
  IF (0.NE.check) THEN
    ALLOCATE(d(mm,nn,kk,s))
    !$OMP PARALLEL DO PRIVATE(i, ix, iy, iz) DEFAULT(NONE) SHARED(d, m, mm, n, nn, k, kk, s)
    DO i = 1, s
      DO ix = 1, mm
        DO iy = 1, nn
          DO iz = 1, kk
            d(ix,iy,iz,i) = 0.0
          END DO
        END DO
      END DO
    END DO

    WRITE(*, "(A)") "Calculating check..."
    !$OMP PARALLEL PRIVATE(i, j, r) DEFAULT(NONE) &
    !$OMP   SHARED(a, dx, dy, dz, d, m, n, k, mm, nn, kk, repetitions)
    ALLOCATE(tm1(mm,n,k), tm2(mm,nn,k))
    tm1 = 0; tm2 = 0;
    DO r = 1, repetitions
      !$OMP DO
      DO i = LBOUND(a, 4), UBOUND(a, 4)
        tm1 = reshape(matmul(dx, reshape(a(:,:,:,i), (/m,n*k/))), (/mm, n, k/))
        DO j = 1, k
          tm2(:,:,j) = matmul(tm1(:,:,j), dy)
        END DO
        ! because we can't reshape d
        d(:,:,:,i) = reshape(matmul(reshape(tm2, (/mm*nn, k/)), dz), (/mm,nn,kk/))
      END DO
    END DO
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2)
    !$OMP END PARALLEL
  END IF

  WRITE(*, "(A)") "Streamed... (BLAS)"
  !$OMP PARALLEL PRIVATE(i, j, r, start) DEFAULT(NONE) &
  !$OMP   SHARED(a, dx, dy, dz, c, m, n, k, mm, nn, kk, duration, repetitions)
  ALLOCATE(tm1(mm,n,k), tm2(mm,nn,k), tm3(mm,nn,kk))
  tm1 = 0; tm2 = 0; tm3 = 3
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP BARRIER
  DO r = 1, repetitions
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL libxsmm_blas_gemm(m=mm, n=n*k, k=m, a=dx, b=a(:,:,1,i), c=tm1(:,:,1), alpha=alpha, beta=beta)
      DO j = 1, k
        CALL libxsmm_blas_gemm(m=mm, n=nn, k=n, a=tm1(:,:,j), b=dy, c=tm2(:,:,j), alpha=alpha, beta=beta)
      END DO
      CALL libxsmm_blas_gemm(m=mm*nn, n=kk, k=k, a=tm2(:,:,1), b=dz, c=tm3(:,:,1), alpha=alpha, beta=beta)
      CALL stream_vector_copy( tm3(1,1,1), c(1,1,1,i), mm*nn*kk )
    END DO
  END DO
  !$OMP BARRIER
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  CALL performance(duration, m, n, k, mm, nn, kk, size)
  IF (check.NE.0) max_diff = MAX(max_diff, validate(c, d))

  WRITE(*, "(A)") "Streamed... (mxm)"
  !$OMP PARALLEL PRIVATE(i, j, r, start) DEFAULT(NONE) &
  !$OMP   SHARED(a, dx, dy, dz, c, m, n, k, mm, nn, kk, duration, repetitions)
  ALLOCATE(tm1(mm,n,k), tm2(mm,nn,k), tm3(mm,nn,kk))
  tm1 = 0; tm2 = 0; tm3 = 3
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP BARRIER
  DO r = 1, repetitions
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL mxmf2(dx, mm, a(:,:,:,i), m, tm1, n*k)
      DO j = 1, k
        CALL mxmf2(tm1(:,:,j), mm, dy, n, tm2(:,:,j), nn)
      END DO
      CALL mxmf2(tm2, mm*nn, dz, k, tm3, kk)
      CALL stream_vector_copy( tm3(1,1,1), c(1,1,1,i), mm*nn*kk )
    END DO
  END DO
  !$OMP BARRIER
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  CALL performance(duration, m, n, k, mm, nn, kk, size)
  IF (check.NE.0) max_diff = MAX(max_diff, validate(c, d))

  WRITE(*, "(A)") "Streamed... (auto-dispatched)"
  !$OMP PARALLEL PRIVATE(i, j, r, start) DEFAULT(NONE) &
  !$OMP   SHARED(a, dx, dy, dz, c, m, n, k, mm, nn, kk, duration, repetitions)
  ALLOCATE(tm1(mm,n,k), tm2(mm,nn,k), tm3(mm,nn,kk))
  tm1 = 0; tm2 = 0; tm3 = 3
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP BARRIER
  DO r = 1, repetitions
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL libxsmm_gemm(m=mm, n=n*k, k=m, a=dx, b=a(:,:,1,i), c=tm1(:,:,1), alpha=alpha, beta=beta)
      DO j = 1, k
        CALL libxsmm_gemm(m=mm, n=nn, k=n, a=tm1(:,:,j), b=dy, c=tm2(:,:,j), alpha=alpha, beta=beta)
      END DO
      CALL libxsmm_gemm(m=mm*nn, n=kk, k=k, a=tm2(:,:,1), b=dz, c=tm3(:,:,1), alpha=alpha, beta=beta)
      CALL stream_vector_copy( tm3(1,1,1), c(1,1,1,i), mm*nn*kk )
    END DO
  END DO
  !$OMP BARRIER
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  CALL performance(duration, m, n, k, mm, nn, kk, size)
  IF (check.NE.0) max_diff = MAX(max_diff, validate(c, d))

  WRITE(*, "(A)") "Streamed... (specialized)"
  CALL libxsmm_dispatch(xmm1, mm, n*k, m, alpha=alpha, beta=beta)
  CALL libxsmm_dispatch(xmm2, mm, nn, n, alpha=alpha, beta=beta)
  CALL libxsmm_dispatch(xmm3, mm*nn, kk, k, alpha=alpha, beta=beta)
  IF (libxsmm_available(xmm1).AND.libxsmm_available(xmm2).AND.libxsmm_available(xmm3)) THEN
    !$OMP PARALLEL PRIVATE(i, j, r, start) !DEFAULT(NONE) SHARED(a, dx, dy, dz, c, m, n, k, mm, nn, kk, duration, repetitions, xmm1, xmm2, xmm3)
    ALLOCATE(tm1(mm,n,k), tm2(mm,nn,k), tm3(mm,nn,kk))
    tm1 = 0; tm2 = 0; tm3 = 3
    !$OMP MASTER
    start = libxsmm_timer_tick()
    !$OMP END MASTER
    !$OMP BARRIER
    DO r = 1, repetitions
      !$OMP DO
      DO i = LBOUND(a, 4), UBOUND(a, 4)
        CALL libxsmm_call(xmm1, C_LOC(dx), C_LOC(a(1,1,1,i)), C_LOC(tm1))
        DO j = 1, k
          CALL libxsmm_call(xmm2, C_LOC(tm1(1,1,j)), C_LOC(dy), C_LOC(tm2(1,1,j)))
        END DO
        CALL libxsmm_call(xmm3, C_LOC(tm2), C_LOC(dz), C_LOC(tm3(1,1,1)))
        CALL stream_vector_copy( tm3(1,1,1), c(1,1,1,i), mm*nn*kk )
      END DO
    END DO
    !$OMP BARRIER
    !$OMP MASTER
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
    !$OMP END MASTER
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL

    CALL performance(duration, m, n, k, mm, nn, kk, size)
    IF (check.NE.0) max_diff = MAX(max_diff, validate(c, d))
  ELSE
    WRITE(*,*) "Could not build specialized function(s)!"
  END IF

  ! Deallocate global arrays
  IF (check.NE.0) DEALLOCATE(d)
  DEALLOCATE(dx, dy, dz)
  DEALLOCATE(a, c)

  ! finalize LIBXSMM
  CALL libxsmm_finalize()

  IF ((0.NE.check).AND.(1.LT.max_diff)) STOP 1

CONTAINS
  FUNCTION validate(ref, test) RESULT(diff)
    REAL(T), DIMENSION(:,:,:,:), intent(in) :: ref, test
    REAL(T) :: diff
    diff = MAXVAL((ref - test) * (ref - test))
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", diff
  END FUNCTION

  SUBROUTINE performance(duration, m, n, k, mm, nn, kk, size)
    DOUBLE PRECISION, INTENT(IN) :: duration
    INTEGER, INTENT(IN)    :: m, n, k, mm, nn, kk
    INTEGER(8), INTENT(IN) :: size
    IF (0.LT.duration) THEN
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:", &
        (size * ((2*m-1)*mm*n*k + mm*(2*n-1)*nn*k + mm*nn*(2*k-1)*kk) * 1D-9 / duration), " GFLOPS/s"
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ", &
        (size * ((m*n*k) + (mm*nn*kk)) * T / (duration * LSHIFT(1_8, 30))), " GB/s"
    END IF
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ", (1D3 * duration)/repetitions, " ms"
  END SUBROUTINE
END PROGRAM
