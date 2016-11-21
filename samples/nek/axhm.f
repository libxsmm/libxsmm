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
  USE, INTRINSIC :: ISO_C_BINDING
  USE :: LIBXSMM
  USE :: STREAM_UPDATE_KERNELS
  !$ USE omp_lib
  IMPLICIT NONE

  INTEGER, PARAMETER :: T = KIND(0D0)
  REAL(T), PARAMETER :: alpha = 1, beta = 0

  REAL(T), ALLOCATABLE, DIMENSION(:,:,:,:), TARGET :: a, c, g1, g2, g3, b, d
  REAL(T), ALLOCATABLE, TARGET :: dx(:,:), dy(:,:), dz(:,:)
  REAL(T), ALLOCATABLE, TARGET, SAVE :: tm1(:,:,:), tm2(:,:,:), tm3(:,:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNMENT :: a, c, g1, g2, g3, d
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNMENT :: tm1, tm2, tm3
  !$OMP THREADPRIVATE(tm1, tm2, tm3)
  TYPE(LIBXSMM_DMMFUNCTION) :: xmm1, xmm2, xmm3
  DOUBLE PRECISION :: duration, max_diff, h1, h2
  INTEGER :: argc, m, n, k, routine, check
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
  IF (4 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(4, argv)
    READ(argv, "(I32)") size1
  ELSE
    size1 = 0
  END IF
  IF (5 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(5, argv)
    READ(argv, "(I32)") size
  ELSE
    size = 0 ! 1 repetition by default
  END IF

  ! Initialize LIBXSMM
  CALL libxsmm_init()

  ! workload is about 2 GByte in memory by default
  size0 = (m * n * k) * T * 6 ! size of a single stream element in Byte
  size1 = MERGE(2048_8, MERGE(size1, ISHFT(ABS(size0 * size1) + ISHFT(1, 20) - 1, -20), 0.LE.size1), 0.EQ.size1)
  size = ISHFT(MERGE(MAX(size, size1), ISHFT(ABS(size) * size0 + ISHFT(1, 20) - 1, -20), 0.LE.size), 20) / size0
  s = ISHFT(size1, 20) / size0; repetitions = size / s; duration = 0; max_diff = 0

  ALLOCATE(a(m,n,k,s))
  ALLOCATE(b(m,n,k,s))
  ALLOCATE(c(m,n,k,s))
  ALLOCATE(g1(m,n,k,s), g2(m,n,k,s), g3(m,n,k,s))
  ALLOCATE(dx(m,m), dy(n,n), dz(k,k))

  ! Initialize
  !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(a, g1, g2, g3, b, c, m, n, k, s)
  DO i = 1, s
    DO ix = 1, m
      DO iy = 1, n
        DO iz = 1, k
          a(ix,iy,iz,i) = ix + iy*m + iz*m*n
          b(ix,iy,iz,i) = -( ix + iy*m + iz*m*n)
          g1(ix,iy,iz,i) = 1.
          g2(ix,iy,iz,i) = 1.
          g3(ix,iy,iz,i) = 1.
          c(ix,iy,iz,i) = 0.
        END DO
      END DO
    END DO
  END DO
  dx = 1.; dy = 1.; dz = 1.
  h1 = 1.; h2 = 1.

  WRITE(*, "(3(A,I0),A,I0,A,I0,A,I0)") "m=", m, " n=", n, " k=", k, &
    " elements=", UBOUND(a, 4), " size=", size1, "MB repetitions=", repetitions

  CALL GETENV("CHECK", argv)
  READ(argv, "(I32)") check
  IF (0.NE.check) THEN
    WRITE(*, "(A)") "Calculating check..."
    ALLOCATE(d(m,n,k,s))
    ! Initialize
    !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(d, m, n, k, s)
    DO i = 1, s
      DO ix = 1, m
        DO iy = 1, n
          DO iz = 1, k
            d(ix,iy,iz,i) = 0.
          END DO
        END DO
      END DO
    END DO

    !$OMP PARALLEL PRIVATE(i, j, r) DEFAULT(NONE) &
    !$OMP   SHARED(a, b, dx, dy, dz, g1, g2, g3, d, m, n, k, h1, h2, repetitions)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m*n,k,1))
    tm1 = 0; tm2 = 0; tm3 = 0
    DO r = 1, repetitions
      !$OMP DO
      DO i = LBOUND(a, 4), UBOUND(a, 4)
        CALL libxsmm_blas_gemm(m=m, n=n*k, k=m, a=dx, b=a(:,:,1,i), c=tm1(:,:,1), alpha=alpha, beta=beta)
        DO j = 1, k
          CALL libxsmm_blas_gemm(m=m, n=n, k=n, a=a(:,:,j,i), b=dy, c=tm2(:,:,j), alpha=alpha, beta=beta)
        END DO
        CALL libxsmm_blas_gemm(m=m*n, n=k, k=k, a=a(:,:,1,i), b=dz, c=tm3(:,:,1), alpha=alpha, beta=beta)
        !DEC$ vector aligned nontemporal
        d(:,:,:,i) = h1*(g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*RESHAPE(tm3, (/m,n,k/))) &
                   + h2*b(:,:,:,i)*a(:,:,:,i)
      END DO
    END DO
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL
  END IF

  c(:,:,:,:) = 0.0
  WRITE(*, "(A)") "Streamed... (BLAS)"
  !$OMP PARALLEL PRIVATE(i, j, r, start) DEFAULT(NONE) &
  !$OMP   SHARED(a, dx, dy, dz, g1, g2, g3, b, c, m, n, k, h1, h2, duration, repetitions)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3 = 0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP BARRIER
  DO r = 1, repetitions
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL libxsmm_blas_gemm(m=m, n=n*k, k=m, a=dx, b=a(:,:,1,i), c=tm1(:,:,1), alpha=alpha, beta=beta)
      DO j = 1, k
        CALL libxsmm_blas_gemm(m=m, n=n, k=n, a=a(:,:,j,i), b=dy, c=tm2(:,:,j), alpha=alpha, beta=beta)
      END DO
      CALL libxsmm_blas_gemm(m=m*n, n=k, k=k, a=a(:,:,1,i), b=dz, c=tm3(:,:,1), alpha=alpha, beta=beta)
      CALL stream_update_helmholtz( g1(1,1,1,i), g2(1,1,1,i), g3(1,1,1,i), &
                                    tm1(1,1,1), tm2(1,1,1), tm3(1,1,1), &
                                    a(1,1,1,i), b(1,1,1,i), c(1,1,1,i), &
                                    h1, h2, m*n*k )
    END DO
  END DO
  !$OMP BARRIER
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  CALL performance(duration, m, n, k, size)
  IF (check.NE.0) max_diff = MAX(max_diff, validate(d, c))

  c(:,:,:,:) = 0.0
  WRITE(*, "(A)") "Streamed... (mxm)"
  !$OMP PARALLEL PRIVATE(i, j, r, start) DEFAULT(NONE) &
  !$OMP   SHARED(a, dx, dy, dz, g1, g2, g3, b, c, m, n, k, h1, h2, duration, repetitions)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3 = 0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP BARRIER
  DO r = 1, repetitions
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL mxmf2(dx, m, a(:,:,:,i), m, tm1, n*k)
      DO j = 1, k
        CALL mxmf2(a(:,:,j,i), m, dy, n, tm2(:,:,j), n)
      END DO
      CALL mxmf2(a(:,:,:,i), m*n, dz, k, tm3, k)
      CALL stream_update_helmholtz( g1(1,1,1,i), g2(1,1,1,i), g3(1,1,1,i), &
                                    tm1(1,1,1), tm2(1,1,1), tm3(1,1,1), &
                                    a(1,1,1,i), b(1,1,1,i), c(1,1,1,i), &
                                    h1, h2, m*n*k )
    END DO
  END DO
  !$OMP BARRIER
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  CALL performance(duration, m, n, k, size)
  IF (check.NE.0) max_diff = MAX(max_diff, validate(d, c))

  c(:,:,:,:) = 0.0
  WRITE(*, "(A)") "Streamed... (auto-dispatched)"
  !$OMP PARALLEL PRIVATE(i, j, r, start) DEFAULT(NONE) &
  !$OMP   SHARED(a, b, dx, dy, dz, g1, g2, g3, c, m, n, k, h1, h2, duration, repetitions)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3 = 0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP BARRIER
  DO r = 1, repetitions
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL libxsmm_gemm(m=m, n=n*k, k=m, a=dx, b=a(:,:,1,i), c=tm1(:,:,1), alpha=alpha, beta=beta)
      DO j = 1, k
        CALL libxsmm_gemm(m=m, n=n, k=n, a=a(:,:,j,i), b=dy, c=tm2(:,:,j), alpha=alpha, beta=beta)
      END DO
      CALL libxsmm_gemm(m=m*n, n=k, k=k, a=a(:,:,1,i), b=dz, c=tm3(:,:,1), alpha=alpha, beta=beta)
      CALL stream_update_helmholtz( g1(1,1,1,i), g2(1,1,1,i), g3(1,1,1,i), &
                                    tm1(1,1,1), tm2(1,1,1), tm3(1,1,1), &
                                    a(1,1,1,i), b(1,1,1,i), c(1,1,1,i), &
                                    h1, h2, m*n*k )
    END DO
  END DO
  !$OMP BARRIER
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  CALL performance(duration, m, n, k, size)
  IF (check.NE.0) max_diff = MAX(max_diff, validate(d, c))

  c(:,:,:,:) = 0.0
  WRITE(*, "(A)") "Streamed... (specialized)"
  CALL libxsmm_dispatch(xmm1, m, n*k, m, alpha=alpha, beta=beta)
  CALL libxsmm_dispatch(xmm2, m, n, n, alpha=alpha, beta=beta)
  CALL libxsmm_dispatch(xmm3, m*n, k, k, alpha=alpha, beta=beta)
  IF (libxsmm_available(xmm1).AND.libxsmm_available(xmm2).AND.libxsmm_available(xmm3)) THEN
    !$OMP PARALLEL PRIVATE(i, j, r, start) !DEFAULT(NONE) SHARED(a, dx, dy, dz, g1, g2, g3, b, c, m, n, k, h1, h2, duration, repetitions, xmm1, xmm2, xmm3)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
    tm1 = 0; tm2 = 0; tm3 = 0
    !$OMP MASTER
    start = libxsmm_timer_tick()
    !$OMP END MASTER
    !$OMP BARRIER
    DO r = 1, repetitions
      !$OMP DO
      DO i = LBOUND(a, 4), UBOUND(a, 4)
        CALL libxsmm_call(xmm1, C_LOC(dx), C_LOC(a(1,1,1,i)), C_LOC(tm1))
        DO j = 1, k
          CALL libxsmm_call(xmm2, C_LOC(a(1,1,j,i)), C_LOC(dy), C_LOC(tm2(1,1,j)))
        END DO
        CALL libxsmm_call(xmm3, C_LOC(a(1,1,1,i)), C_LOC(dz), C_LOC(tm3))
        CALL stream_update_helmholtz( g1(1,1,1,i), g2(1,1,1,i), g3(1,1,1,i), &
                                      tm1(1,1,1), tm2(1,1,1), tm3(1,1,1), &
                                      a(1,1,1,i), b(1,1,1,i), c(1,1,1,i), &
                                      h1, h2, m*n*k )
      END DO
    END DO
    !$OMP BARRIER
    !$OMP MASTER
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
    !$OMP END MASTER
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL

    ! Print Performance Summary and check results
    CALL performance(duration, m, n, k, size)
    IF (check.NE.0) max_diff = MAX(max_diff, validate(d, c))
  ELSE
    WRITE(*,*) "Could not build specialized function(s)!"
  END IF

  ! Deallocate global arrays
  IF (0.NE.check) DEALLOCATE(d)
  DEALLOCATE(dx, dy, dz)
  DEALLOCATE(g1, g2, g3)
  DEALLOCATE(a, b, c)

  ! finalize LIBXSMM
  CALL libxsmm_finalize()

  IF ((0.NE.check).AND.(1.LT.max_diff)) STOP 1

CONTAINS
  FUNCTION validate(ref, test) RESULT(diff)
    REAL(T), DIMENSION(:,:,:,:), INTENT(IN) :: ref, test
    REAL(T) :: diff
    diff = MAXVAL((ref - test) * (ref - test))
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", diff
  END FUNCTION

  SUBROUTINE performance(duration, m, n, k, size)
    DOUBLE PRECISION, INTENT(IN) :: duration
    INTEGER, INTENT(IN)    :: m, n, k
    INTEGER(8), INTENT(IN) :: size
    IF (0.LT.duration) THEN
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:", &
        (size * m * n * k * (2*(m+n+k) + 2 + 4) * 1D-9 / duration), " GFLOPS/s"
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ", &
        (size * m * n * k * (6) * T / (duration * ISHFT(1_8, 30))), " GB/s"
    END IF
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ", (1D3 * duration)/repetitions, " ms"
  END SUBROUTINE
END PROGRAM
