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


PROGRAM grad
  USE :: LIBXSMM

  !$ USE omp_lib
  IMPLICIT NONE

  INTEGER, PARAMETER :: T = LIBXSMM_FLD_KIND
  REAL(T), PARAMETER :: alpha = 1, beta = 0

  REAL(T), allocatable, dimension(:,:,:,:), target :: a, cx, cy, cz
  REAL(T), allocatable, dimension(:,:,:,:), target :: rx, ry, rz
  REAL(T), allocatable, target :: dx(:,:), dy(:,:), dz(:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNMENT :: a, cx, cy, cz
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNMENT :: rx, ry, rz 
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

  ALLOCATE(cx(m,n,k,s), cy(m,n,k,s), cz(m,n,k,s))
  ALLOCATE(a(m,n,k,s))
  ALLOCATE(dx(m,m), dy(n,n), dz(k,k))

  ! Initialize 
  !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(a, cx, cy, cz, m, n, k, s)
  DO i = 1, s
    DO ix = 1, m
      DO iy = 1, n
        DO iz = 1, k
          a(ix,iy,iz,i) = ix + iy*m + iz*m*n
          cx(ix,iy,iz,i) = 0.
          cy(ix,iy,iz,i) = 0.
          cz(ix,iy,iz,i) = 0.
        END DO
      END DO
    END DO
  END DO 
  dx = 1.; dy = 2.; dz = 3.

  WRITE(*, "(A,I0,A,I0,A,I0,A,I0)") "m=", m, " n=", n, " k=", k, " size=", UBOUND(a, 4) 

  CALL GETENV("CHECK", argv)
  READ(argv, "(I32)") check
  IF (0.NE.check) THEN
    ALLOCATE(rx(m,n,k,s), ry(m,n,k,s), rz(m,n,k,s))
    !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(rx, ry, rz, m, n, k, s)
    DO i = 1, s
      DO ix = 1, m
        DO iy = 1, n
          DO iz = 1, k
            rx(ix,iy,iz,i) = 0.
            ry(ix,iy,iz,i) = 0.
            rz(ix,iy,iz,i) = 0.
          END DO
        END DO
      END DO
    END DO 

    WRITE(*, "(A)") "Calculating check..."
    !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) &
    !$OMP   SHARED(duration, a, dx, dy, dz, rx, ry, rz, m, n, k)
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      rx(:,:,:,i) = reshape( matmul(dx, reshape(a(:,:,:,i), (/m,n*k/))), (/m,n,k/))
      do j = 1, k
          ry(:,:,j,i) = matmul(a(:,:,j,i), dy)
      enddo
      rz(:,:,:,i) = reshape( matmul(reshape(a(:,:,:,i), (/m*n,k/)), dz), (/m,n,k/))
    END DO
    ! Deallocate thread-local arrays
    !$OMP END PARALLEL
  END IF

  WRITE(*, "(A)") "Streamed... (mxm)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) &
  !$OMP   SHARED(duration, a, dx, dy, dz, cx, cy, cz, m, n, k)
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    call mxmf2(dx, m, a(:,:,:,i), m, cx(:,:,:,i), n*k)
    do j = 1, k
        call mxmf2(a(:,:,j,i), m, dy, n, cy(:,:,j,i), n)
    enddo
    call mxmf2(a(:,:,:,i), m*n, dz, k, cz(:,:,:,i), k)
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  call performance(duration, m, n, k, s)
  if (check.NE.0) call validate(rx, ry, rz, cx, cy, cz)

  WRITE(*, "(A)") "Streamed... (auto-dispatched)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) &
  !$OMP   SHARED(duration, a, dx, dy, dz, cx, cy, cz, m, n, k)
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    call libxsmm_mm(m, n*k, m, dx, a(:,:,1,i), cx(:,:,1,i), LIBXSMM_FLAGS, alpha, beta)
    do j = 1, k
        call libxsmm_mm(m, n, n, a(:,:,j,i), dy, cy(:,:,j,i), LIBXSMM_FLAGS, alpha, beta)
    enddo
    call libxsmm_mm(m*n, k, k, a(:,:,1,i), dz, cz(:,:,1,i), LIBXSMM_FLAGS, alpha, beta)
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  call performance(duration, m, n, k, s)
  if (check.NE.0) call validate(rx, ry, rz, cx, cy, cz)

  WRITE(*, "(A)") "Streamed... (specialized)"
  CALL libxsmm_dispatch(xmm1, m, n*k, m, alpha, beta)
  CALL libxsmm_dispatch(xmm2, m, n, n, alpha, beta)
  CALL libxsmm_dispatch(xmm3, m*n, k, k, alpha, beta)
  IF (libxsmm_available(xmm1).AND.libxsmm_available(xmm2).AND.libxsmm_available(xmm3)) THEN
  !$OMP PARALLEL PRIVATE(i, start) !DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, cx, cy, cz, m, n, k, xmm1, xmm2, xmm3)
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    CALL libxsmm_call(xmm1,  C_LOC(dx), C_LOC(a(1,1,1,i)), C_LOC(cx(1,1,1,i)))
    do j = 1, k
        call libxsmm_call(xmm2, C_LOC(a(1,1,j,i)), C_LOC(dy), C_LOC(cy(1,1,j,i)))
    enddo
    CALL libxsmm_call(xmm3, C_LOC(a(1,1,1,i)), C_LOC(dz), C_LOC(cz(1,1,1,i)))
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  call performance(duration, m, n, k, s)
  if (check.NE.0) call validate(rx, ry, rz, cx, cy, cz)
  ELSE
    WRITE(*,*) "Could not build specialized function(s)!"
  END IF

  ! Deallocate global arrays
  DEALLOCATE(a)
  deallocate(dx, dy, dz)
  DEALLOCATE(cx, cy, cz)
  IF (0.NE.check) THEN
    DEALLOCATE(rx, ry, rz)
  END IF

contains
  SUBROUTINE validate(refx, refy, refz, testx, testy, testz)
    REAL(T), DIMENSION(:,:,:,:), INTENT(IN) :: refx, refy, refz
    REAL(T), DIMENSION(:,:,:,:), INTENT(IN) :: testx, testy, testz
    real(T) :: diff

    diff = MAXVAL((refx - testx) * (refx - testx))
    diff = MAX(MAXVAL((refy - testy) * (refy - testy)), diff)
    diff = MAX(MAXVAL((refz - testz) * (refz - testz)), diff)

    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", diff 
  END SUBROUTINE validate

  SUBROUTINE performance(duration, m, n, k, s)
    REAL(8), INTENT(IN)    :: duration
    INTEGER, INTENT(IN)    :: m, n, k
    INTEGER(8), INTENT(IN) :: s

    IF (0.LT.duration) THEN
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:", &
        (s * m * n * k * (2*(m+n+k) - 3) * 1D-9 / duration), " GFLOPS/s"
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ", &
        (s * m * n * k * (4) * T / (duration * ISHFT(1_8, 30))), " GB/s"
    ENDIF
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ", 1D3 * duration, " ms"
  END SUBROUTINE
END PROGRAM
