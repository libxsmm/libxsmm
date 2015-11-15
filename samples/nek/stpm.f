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
!* Hans Pabst (Intel Corp.), Alexander Heinecke (Intel Corp.), and           *! 
!* Maxwell Hutchinson (University of Chicago)                                *!
!*****************************************************************************!


PROGRAM stpm
  USE :: LIBXSMM

  !$ USE omp_lib
  IMPLICIT NONE

  INTEGER, PARAMETER :: T = LIBXSMM_FLD_KIND
  REAL(T), PARAMETER :: alpha = 1, beta = 0

  REAL(T), allocatable, dimension(:,:,:,:), target :: a, c, g1, g2, g3, d
  REAL(T), allocatable, target :: dx(:,:), dy(:,:), dz(:,:)
  REAL(T), ALLOCATABLE, TARGET, SAVE :: tm1(:,:,:), tm2(:,:,:), tm3(:,:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNMENT :: a, c, g1, g2, g3, d
  !$OMP THREADPRIVATE(tm1, tm2, tm3)
  TYPE(LIBXSMM_DMM_FUNCTION) :: xmm1, xmm2, xmm3
  INTEGER :: argc, m, n, k, routine, check
  INTEGER(8) :: i, j, s, ix, iy, iz, start
  CHARACTER(32) :: argv
  REAL(8) :: duration

  duration = 0

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
    READ(argv, "(I32)") i
  ELSE
    i = 2 ! 2 GByte for A and B (and C, but this currently not used by the F90 test)
  END IF
  s = ISHFT(MAX(i, 0_8), 30) / ((m * n * k) * T * 5)

  ALLOCATE(c(m,n,k,s))
  ALLOCATE(a(m,n,k,s))
  ALLOCATE(g1(m,n,k,s), g2(m,n,k,s), g3(m,n,k,s))
  ALLOCATE(dx(m,m), dy(n,n), dz(k,k))

  ! Initialize 
  !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(a, g1, g2, g3, c, m, n, k, s)
  DO i = 1, s
    DO ix = 1, m
      DO iy = 1, n
        DO iz = 1, k
          a(ix,iy,iz,i) = ix + iy*m + iz*m*n
          g1(ix,iy,iz,i) = 1.
          g2(ix,iy,iz,i) = 1.
          g3(ix,iy,iz,i) = 1.
          c(ix,iy,iz,i) = 0.
        END DO
      END DO
    END DO
  END DO 
  dx = 1.; dy = 1.; dz = 1.

  WRITE(*, "(A,I0,A,I0,A,I0,A,I0)") "m=", m, " n=", n, " k=", k, " size=", UBOUND(a, 4) 

  CALL GETENV("CHECK", argv)
  READ(argv, "(I32)") check
  IF (0.NE.check) THEN
    ALLOCATE(d(m,n,k,s))
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

    WRITE(*, "(A)") "Calculating check..."
    !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) &
    !$OMP   SHARED(duration, a, dx, dy, dz, g1, g2, g3, d, m, n, k)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
    tm1 = 0; tm2 = 0; tm3 = 0
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      tm1 = reshape( matmul(dx, reshape(a(:,:,:,i), (/m,n*k/))), (/m,n,k/))
      DO j = 1, k
          tm2(:,:,j) = matmul(a(:,:,j,i), dy)
      END DO
      tm3 = reshape( matmul(reshape(a(:,:,:,i), (/m*n,k/)), dz), (/m,n,k/))
      !DEC$ vector aligned nontemporal
      d(:,:,:,i) = g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*tm3
    END DO
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL
  END IF

  WRITE(*, "(A)") "Streamed... (mxm)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) &
  !$OMP   SHARED(duration, a, dx, dy, dz, g1, g2, g3, c, m, n, k)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3 = 0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    CALL mxmf2(dx, m, a(:,:,:,i), m, tm1, n*k)
    DO j = 1, k
        CALL mxmf2(a(:,:,j,i), m, dy, n, tm2(:,:,j), n)
    END DO
    CALL mxmf2(a(:,:,:,i), m*n, dz, k, tm3, k)
    !DEC$ vector aligned nontemporal
    c(:,:,:,i) = g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*tm3
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  CALL performance(duration, m, n, k, s)
  IF (check.NE.0) CALL validate(d, c)

  WRITE(*, "(A)") "Streamed... (auto-dispatched)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) &
  !$OMP   SHARED(duration, a, dx, dy, dz, g1, g2, g3, c, m, n, k)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3 = 0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    CALL libxsmm_mm(m, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1), alpha, beta)
    DO j = 1, k
        CALL libxsmm_mm(m, n, n, a(:,:,j,i), dy, tm2(:,:,j), alpha, beta)
    END DO
    CALL libxsmm_mm(m*n, k, k, reshape(a(:,:,:,i), (/m*n,k/)), dz, tm3(:,:,1), alpha, beta)
    !DEC$ vector aligned nontemporal
    c(:,:,:,i) = g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*tm3
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  CALL performance(duration, m, n, k, s)
  IF (check.NE.0) CALL validate(d, c)

  WRITE(*, "(A)") "Streamed... (specialized)"
  CALL libxsmm_dispatch(xmm1, m, n*k, m, alpha, beta)
  CALL libxsmm_dispatch(xmm2, m, n, n, alpha, beta)
  CALL libxsmm_dispatch(xmm3, m*n, k, k, alpha, beta)
  IF (libxsmm_available(xmm1).AND.libxsmm_available(xmm2).AND.libxsmm_available(xmm3)) THEN
    !$OMP PARALLEL PRIVATE(i, start) !DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, g1, g2, g3, c, m, n, k, xmm1, xmm2, xmm3)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
    tm1 = 0; tm2 = 0; tm3 = 0
    !$OMP MASTER
    start = libxsmm_timer_tick()
    !$OMP END MASTER
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL libxsmm_call(xmm1, C_LOC(dx), C_LOC(a(:,:,:,i)), C_LOC(tm1(:,:,1)))
      DO j = 1, k
          CALL libxsmm_call(xmm2, C_LOC(a(:,:,j,i)), C_LOC(dy), C_LOC(tm2(:,:,j)))
      END DO
      CALL libxsmm_call(xmm3, C_LOC(a(:,:,:,i)), C_LOC(dz), C_LOC(tm3(:,:,1)))
      !DEC$ vector aligned nontemporal
      c(:,:,:,i) = g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*tm3
    END DO
    !$OMP MASTER
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
    !$OMP END MASTER
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL

    ! Print Performance Summary and check results
    CALL performance(duration, m, n, k, s)
    IF (check.NE.0) CALL validate(d, c)
  ELSE
    WRITE(*,*) "Could not build specialized function(s)!"
  END IF

  ! Deallocate global arrays
  DEALLOCATE(a)
  deallocate(g1, g2, g3)
  deallocate(dx, dy, dz)
  DEALLOCATE(c)
  IF (0.NE.check) THEN
    DEALLOCATE(d)
  END IF

contains
  SUBROUTINE validate(ref, test)
    REAL(T), DIMENSION(:,:,:,:), INTENT(IN) :: ref, test

    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", MAXVAL((ref - test) * (ref - test))
  END SUBROUTINE validate

  SUBROUTINE performance(duration, m, n, k, s)
    REAL(8), INTENT(IN)    :: duration
    INTEGER, INTENT(IN)    :: m, n, k
    INTEGER(8), INTENT(IN) :: s

    IF (0.LT.duration) THEN
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:", &
        (s * m * n * k * (2*(m+n+k) + 2 + 4) * 1D-9 / duration), " GFLOPS/s"
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ", &
        (s * m * n * k * (5) * T / (duration * ISHFT(1_8, 30))), " GB/s"
    END IF
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ", 1D3 * duration, " ms"
  END SUBROUTINE
END PROGRAM
