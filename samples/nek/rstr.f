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

  REAL(T), allocatable, dimension(:,:,:,:), target :: a, c, d
  REAL(T), allocatable, target :: dx(:,:), dy(:,:), dz(:,:)
  REAL(T), ALLOCATABLE, TARGET, SAVE :: tm1(:,:,:), tm2(:,:,:), tm3(:,:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNED_MAX :: a, c, d
  !$OMP THREADPRIVATE(tm1, tm2, tm3)
  PROCEDURE(LIBXSMM_DMM_FUNCTION), POINTER :: xmm1, xmm2, xmm3
  TYPE(LIBXSMM_DGEMM_XARGS) :: xargs
  INTEGER :: argc, m, n, k, routine, check
  integer :: mm, nn, kk
  INTEGER(8) :: i, j, s, ix, iy, iz, start
  CHARACTER(32) :: argv
  TYPE(C_FUNPTR) :: f1, f2, f3
  REAL(8) :: duration

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
    READ(argv, "(I32)") mm
  ELSE
    mm = 10
  END IF
  IF (5 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(5, argv)
    READ(argv, "(I32)") nn
  ELSE
    nn = mm
  END IF
  IF (6 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(6, argv)
    READ(argv, "(I32)") kk
  ELSE
    kk = mm
  END IF
  IF (7 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(7, argv)
    READ(argv, "(I32)") routine
  ELSE
    routine = -1
  END IF
  IF (8 <= argc) THEN
    CALL GET_COMMAND_ARGUMENT(8, argv)
    READ(argv, "(I32)") i
  ELSE
    i = 2 ! 2 GByte for A and B (and C, but this currently not used by the F90 test)
  END IF
  s = ISHFT(MAX(i, 0_8), 29) / (((m * n * k) + (nn * mm * kk)) * T)

  ALLOCATE(a(m,n,k,s))
  ALLOCATE(c(mm,nn,kk,s))
  ALLOCATE(dx(mm,m), dy(n,nn), dz(k,kk))

  ! Initialize
  !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(a, m, mm, n, nn, k, kk, s)
  DO i = 1, s
    do ix = 1, m
      do iy = 1, n
        do iz = 1, k
          a(ix,iy,iz,i) = ix + iy*m + iz*m*n
        enddo
      enddo
    enddo
  END DO 
  !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(c, m, mm, n, nn, k, kk, s)
  DO i = 1, s
    do ix = 1, mm
      do iy = 1, nn
        do iz = 1, kk
          c(ix,iy,iz,i) = 0 
        enddo
      enddo
    enddo
  END DO 
  dx = 1.
  dy = 1.
  dz = 1.

  WRITE(*, "(6(A,I0),A,I0)") "m=", m, " n=", n, " k=", k, " mm=", mm, " nn=", nn, " kk=", kk, " size=", UBOUND(a, 4) 

  CALL GETENV("CHECK", argv)
  READ(argv, "(I32)") check
  IF (0.NE.check) THEN
    ALLOCATE(d(mm,nn,kk,s))
    d = 0

    WRITE(*, "(A)") "Calculating check..."
    !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) &
    !$OMP   SHARED(duration, xargs, a, dx, dy, dz, d, m, n, k, mm, nn, kk)
    ALLOCATE(tm1(mm,n,k), tm2(mm,nn,k))
    tm1 = 0; tm2 = 0;
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      tm1 = reshape(matmul(dx, reshape(a(:,:,:,i), (/m,n*k/))), (/mm, n, k/))
      do j = 1, k
          tm2(:,:,j) = matmul(tm1(:,:,j), dy)
      enddo
      ! because we can't reshape d
      d(:,:,:,i) = reshape(matmul(reshape(tm2, (/mm*nn, k/)), dz), (/mm,nn,kk/))
    END DO
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2)
    !$OMP END PARALLEL
  END IF

  WRITE(*, "(A)") "Streamed... (auto-dispatched)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) &
  !$OMP   SHARED(duration, xargs, a, dx, dy, dz, c, m, n, k, mm, nn, kk)
  ALLOCATE(tm1(mm,n,k), tm2(mm,nn,k))
  tm1 = 0; tm2 = 0;
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    call libxsmm_mm(mm, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1), xargs)
    do j = 1, k
        call libxsmm_mm(mm, nn, n, tm1(:,:,j), dy, tm2(:,:,j), xargs)
    enddo
    call libxsmm_mm(mm*nn, kk, k, reshape(tm2, (/mm*nn,k/)), dz, c(:,:,1,i), xargs)
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2)
  !$OMP END PARALLEL

  call performance(duration, m, n, k, mm, nn, kk, s)
  if (check.NE.0) call validate(c, d)

  WRITE(*, "(A)") "Streamed... (mxm)"
  !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) &
  !$OMP   SHARED(duration, xargs, a, dx, dy, dz, c, m, n, k, mm, nn, kk)
  ALLOCATE(tm1(mm,n,k), tm2(mm,nn,k))
  tm1 = 0; tm2 = 0;
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    call mxmf2(dx, mm, a(:,:,:,i), m, tm1, n*k, xargs)
    do j = 1, k
        call mxmf2(tm1(:,:,j), mm, dy, n, tm2(:,:,j), nn, xargs)
    enddo
    call mxmf2(tm2, mm*nn, dz, k, c(:,:,:,i), kk, xargs)
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2)
  !$OMP END PARALLEL

  call performance(duration, m, n, k, mm, nn, kk, s)
  if (check.NE.0) call validate(c, d)

  WRITE(*, "(A)") "Streamed... (specialized)"
  f1 = libxsmm_dispatch(mm, n*k, m, alpha, beta)
  f2 = libxsmm_dispatch(mm, nn, n, alpha, beta)
  f3 = libxsmm_dispatch(mm*nn, kk, k, alpha, beta)
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
  !$OMP PARALLEL PRIVATE(i, start) !DEFAULT(NONE) SHARED(duration, xargs, a, dx, dy, dz, c, m, n, k, mm, nn, kk, xmm1, xmm2, xmm3)
  ALLOCATE(tm1(mm,n,k), tm2(mm,nn,k))
  tm1 = 0; tm2 = 0
  !$OMP MASTER
  start = libxsmm_timer_tick()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    CALL xmm1(dx, a(1,1,1,i), tm1, xargs)
    do j = 1, k
        call xmm2(tm1(1,1,j), dy, tm2(1,1,j), xargs)
    enddo
    CALL xmm3(tm2, dz, c(1,1,1,i), xargs)
  END DO
  !$OMP MASTER
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2)
  !$OMP END PARALLEL

  call performance(duration, m, n, k, mm, nn, kk, s)
  if (check.NE.0) call validate(c, d)

  ! Deallocate global arrays
  DEALLOCATE(a)
  DEALLOCATE(dx, dy, dz)
  DEALLOCATE(c)

contains
  subroutine validate(ref, test)
    real(T), dimension(:,:,:,:), intent(in) :: ref, test

    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", MAXVAL((ref - test) * (ref - test))
  end subroutine validate

  subroutine performance(duration, m, n, k, mm, nn, kk, s)
    real(8), intent(in) :: duration
    integer, intent(in) :: m, n, k, mm, nn, kk
    integer(8), intent(in) :: s

    IF (0.LT.duration) THEN
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:", &
        (s * ((2*m-1)*mm*n*k + mm*(2*n-1)*nn*k + mm*nn*(2*k-1)*kk) * 1D-9 / duration), " GFLOPS/s"
      WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ", &
        (s * (m * n * k + mm*nn*kk) * T / (duration * LSHIFT(1_8, 30))), " GB/s"
    ENDIF
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ", 1D3 * duration, " ms"
  end subroutine performance

END PROGRAM
