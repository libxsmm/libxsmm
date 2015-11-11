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
  USE, INTRINSIC :: ISO_C_BINDING
  USE :: LIBXSMM
  USE :: STREAM_UPDATE_KERNELS
  !$ USE omp_lib
  IMPLICIT NONE

  INTEGER, PARAMETER :: T = LIBXSMM_FLD_KIND
  REAL(T), PARAMETER :: alpha = 1, beta = 0

  REAL(T), allocatable, dimension(:,:,:,:), target :: a, c, g1, g2, g3, b, d
  REAL(T), allocatable, target :: dx(:,:), dy(:,:), dz(:,:)
  REAL(T), ALLOCATABLE, TARGET, SAVE :: tm1(:,:,:), tm2(:,:,:), tm3(:,:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNMENT :: a, c, g1, g2, g3, d
  !$OMP THREADPRIVATE(tm1, tm2, tm3)
  PROCEDURE(LIBXSMM_DMM_FUNCTION), POINTER :: xmm1, xmm2, xmm3
  TYPE(LIBXSMM_DGEMM_XARGS) :: xargs
  INTEGER :: argc, m, n, k, routine, check
  INTEGER(8) :: i, j, s, ix, iy, iz, start
  CHARACTER(32) :: argv
  TYPE(C_FUNPTR) :: f1, f2, f3
  REAL(8) :: duration, h1, h2

  xargs = LIBXSMM_DGEMM_XARGS_CTOR(alpha, beta)
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
  s = ISHFT(MAX(i, 0_8), 30) / ((m * n * k) * T * 6)

  ALLOCATE(a(m,n,k,s))
  ALLOCATE(b(m,n,k,s))
  ALLOCATE(c(m,n,k,s))
  ALLOCATE(g1(m,n,k,s), g2(m,n,k,s), g3(m,n,k,s))
  ALLOCATE(dx(m,m), dy(n,n), dz(k,k))

  ! Initialize 
  !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(a, g1, g2, g3, b, c, m, n, k, s)
  DO i = 1, s
    do ix = 1, m
      do iy = 1, n
        do iz = 1, k
          a(ix,iy,iz,i) = ix + iy*m + iz*m*n
          b(ix,iy,iz,i) = -( ix + iy*m + iz*m*n)
          g1(ix,iy,iz,i) = 1.
          g2(ix,iy,iz,i) = 1.
          g3(ix,iy,iz,i) = 1.
          c(ix,iy,iz,i) = 0.
        enddo
      enddo
    enddo
  END DO 
  dx = 1.; dy = 1.; dz = 1.
  h1 = 1.; h2 = 1.

  WRITE(*, "(A,I0,A,I0,A,I0,A,I0)") "m=", m, " n=", n, " k=", k, " size=", UBOUND(a, 4) 

  CALL GETENV("CHECK", argv)
  READ(argv, "(I32)") check
  IF (0.NE.check) THEN
    WRITE(*, "(A)") "Calculating check..."
    ALLOCATE(d(m,n,k,s))
    d = 0

    !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) &
    !$OMP   SHARED(duration, xargs, a, b, dx, dy, dz, g1, g2, g3, d, m, n, k, h1, h2)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m*n,k,1))
    tm1 = 0; tm2 = 0; tm3=0
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      call libxsmm_blasmm(m, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1), xargs)
      do j = 1, k
          call libxsmm_blasmm(m, n, n, a(:,:,j,i), dy, tm2(:,:,j), xargs)
      enddo
      call libxsmm_blasmm(m*n, k, k, reshape(a(:,:,:,i), (/m*n,k/)), dz, tm3(:,:,1), xargs)
      !DEC$ vector aligned nontemporal
      d(:,:,:,i) = h1*(g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*reshape(tm3, (/m,n,k/))) &
                 + h2*b(:,:,:,i)*a(:,:,:,i)
    END DO
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL
  END IF

  c(:,:,:,:) = 0.0
  WRITE(*, "(A)") "Streamed... (BLAS)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) &
  !$OMP   SHARED(duration, xargs, a, dx, dy, dz, g1, g2, g3, b, c, m, n, k, h1, h2)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3=0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    call libxsmm_blasmm(m, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1), xargs)
    do j = 1, k
        call libxsmm_blasmm(m, n, n, a(:,:,j,i), dy, tm2(:,:,j), xargs)
    enddo
    call libxsmm_blasmm(m*n, k, k, reshape(a(:,:,:,i), (/m*n,k/)), dz, tm3(:,:,1), xargs)
    CALL stream_update_helmholtz( g1(1,1,1,i), g2(1,1,1,i), g3(1,1,1,i), &
                                  tm1(1,1,1), tm2(1,1,1), tm3(1,1,1), &
                                  a(1,1,1,i), b(1,1,1,i), c(1,1,1,i), &
                                  h1, h2, m*n*k )
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  call performance(duration, m, n, k, s)
  if (check.NE.0) call validate(d, c)

  c(:,:,:,:) = 0.0
  WRITE(*, "(A)") "Streamed... (mxm)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) &
  !$OMP   SHARED(duration, xargs, a, dx, dy, dz, g1, g2, g3, b, c, m, n, k, h1, h2)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3=0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    CALL mxmf2(dx, m, a(:,:,:,i), m, tm1, n*k, xargs)
    do j = 1, k
        CALL mxmf2(a(:,:,j,i), m, dy, n, tm2(:,:,j), n, xargs)
    enddo
    CALL mxmf2(a(:,:,:,i), m*n, dz, k, tm3, k, xargs)
    CALL stream_update_helmholtz( g1(1,1,1,i), g2(1,1,1,i), g3(1,1,1,i), &
                                  tm1(1,1,1), tm2(1,1,1), tm3(1,1,1), &
                                  a(1,1,1,i), b(1,1,1,i), c(1,1,1,i), &
                                  h1, h2, m*n*k )
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  call performance(duration, m, n, k, s)
  if (check.NE.0) call validate(d, c)

  c(:,:,:,:) = 0.0
  WRITE(*, "(A)") "Streamed... (auto-dispatched)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) &
  !$OMP   SHARED(duration, xargs, a, b, dx, dy, dz, g1, g2, g3, c, m, n, k, h1, h2)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3=0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    CALL libxsmm_mm(m, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1), xargs)
    do j = 1, k
        CALL libxsmm_mm(m, n, n, a(:,:,j,i), dy, tm2(:,:,j), xargs)
    enddo
    CALL libxsmm_mm(m*n, k, k, reshape(a(:,:,:,i), (/m*n,k/)), dz, tm3(:,:,1), xargs)
    CALL stream_update_helmholtz( g1(1,1,1,i), g2(1,1,1,i), g3(1,1,1,i), &
                                  tm1(1,1,1), tm2(1,1,1), tm3(1,1,1), &
                                  a(1,1,1,i), b(1,1,1,i), c(1,1,1,i), &
                                  h1, h2, m*n*k )
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  call performance(duration, m, n, k, s)
  if (check.NE.0) call validate(d, c)

  c(:,:,:,:) = 0.0
  WRITE(*, "(A)") "Streamed... (specialized)"
  f1 = libxsmm_dispatch(m, n*k, m, alpha, beta)
  f2 = libxsmm_dispatch(m, n, n, alpha, beta)
  f3 = libxsmm_dispatch(m*n, k, k, alpha, beta)
  if (C_ASSOCIATED(f1)) then
    CALL C_F_PROCPOINTER(f1, xmm1)
  else
    write(*,*) "f1 not built"
  endif
  if (C_ASSOCIATED(f2)) then
    CALL C_F_PROCPOINTER(f2, xmm2)
  else
    write(*,*) "f2 not built"
  endif
  if (C_ASSOCIATED(f3)) then
    CALL C_F_PROCPOINTER(f3, xmm3)
  else
    write(*,*) "f3 not built"
  endif
  !$OMP PARALLEL PRIVATE(i, start) !DEFAULT(NONE) SHARED(duration, xargs, a, dx, dy, dz, g1, g2, g3, b, c, m, n, k, xmm1, xmm2, xmm3, h1, h2)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3=0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    CALL xmm1(dx, a(1,1,1,i), tm1, xargs)
    do j = 1, k
        CALL xmm2(a(1,1,j,i), dy, tm2(1,1,j), xargs)
    enddo
    CALL xmm3(a(1,1,1,i), dz, tm3, xargs)
    CALL stream_update_helmholtz( g1(1,1,1,i), g2(1,1,1,i), g3(1,1,1,i), &
                                  tm1(1,1,1), tm2(1,1,1), tm3(1,1,1), &
                                  a(1,1,1,i), b(1,1,1,i), c(1,1,1,i), &
                                  h1, h2, m*n*k )
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  ! Print Performance Summary and check results
  call performance(duration, m, n, k, s)
  if (check.NE.0) call validate(d, c)

  ! Deallocate global arrays
  DEALLOCATE(a)
  DEALLOCATE(b)
  deallocate(g1, g2, g3)
  deallocate(dx, dy, dz)
  DEALLOCATE(c)
  IF (0.NE.check) THEN
    DEALLOCATE(d)
  END IF

CONTAINS
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
        (s * m * n * k * (6) * T / (duration * ISHFT(1_8, 30))), " GB/s"
    ENDIF
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ", 1D3 * duration, " ms"
  END SUBROUTINE
END PROGRAM
