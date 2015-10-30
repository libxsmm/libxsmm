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

  INTEGER, PARAMETER :: T = LIBXSMM_DOUBLE_PRECISION

  REAL(T), allocatable, dimension(:,:,:,:), target :: a, c, g1, g2, g3, d
  real(T), allocatable, target :: dx(:,:), dy(:,:), dz(:,:)
  REAL(T), ALLOCATABLE, TARGET, SAVE :: tm1(:,:,:), tm2(:,:,:), tm3(:,:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNED_MAX :: a, c, g1, g2, g3, d
  !$OMP THREADPRIVATE(tm1, tm2, tm3)
  PROCEDURE(LIBXSMM_DMM_FUNCTION), POINTER :: dmm1, dmm2, dmm3
  INTEGER :: argc, m, n, k, routine, check
  INTEGER(8) :: i, j, s, ix, iy, iz
  CHARACTER(32) :: argv
  TYPE(C_FUNPTR) :: f1, f2, f3

  REAL(8) :: duration

  duration = 0

  argc = IARGC()
  IF (1 <= argc) THEN
    CALL GETARG(1, argv)
    READ(argv, "(I32)") m
  ELSE
    m = 8
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
  s = LSHIFT(INT8(MAX(i, 0)), 30) / ((m * n * k) * T * 5)

  ALLOCATE(c(m,n,k,s))
  ALLOCATE(a(m,n,k,s))
  ALLOCATE(g1(m,n,k,s), g2(m,n,k,s), g3(m,n,k,s))
  ALLOCATE(dx(m,m), dy(n,n), dz(k,k))

  ! Initialize 
  !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(a, g1, g2, g3, c, m, n, k, s)
  DO i = 1, s
    do ix = 1, m
      do iy = 1, n
        do iz = 1, k
          a(ix,iy,iz,i) = ix + iy*m + iz*m*n
          g1(ix,iy,iz,i) = 1.
          g2(ix,iy,iz,i) = 1.
          g3(ix,iy,iz,i) = 1.
          c(ix,iy,iz,i) = 0.
        enddo
      enddo
    enddo
  END DO 
  dx = 1.; dy = 1.; dz = 1.

  WRITE (*, "(A,I3,A,I3,A,I3,A,I10)") "m=", m, " n=", n, " k=", k, " size=", UBOUND(a, 4) 

  CALL GETENV("CHECK", argv)
  READ(argv, "(I32)") check
  IF (0.NE.check) THEN
    ALLOCATE(d(m,n,k,s))
    d = 0.
    WRITE(*, "(A)") "Calculating check..."
    !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, g1, g2, g3, d, m, n, k, f1, f2, f3)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m*n,k,1))
    tm1 = 0; tm2 = 0; tm3=0
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      call libxsmm_blasmm(m, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1))
      do j = 1, k
          call libxsmm_blasmm(m, n, n, a(:,:,j,i), dy, tm2(:,:,j))
      enddo
      call libxsmm_blasmm(m*n, k, k, reshape(a(:,:,:,i), (/m*n,k/)), dz, tm3(:,:,1))
      !DEC$ vector aligned nontemporal
      d(:,:,:,i) = g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*reshape(tm3, (/m,n,k/))
    END DO
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL
  END IF

  WRITE(*, "(A)") "Streamed... (compiled)"
  !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, g1, g2, g3, c, m, n, k, f1, f2, f3)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3=0
  !$OMP MASTER
  !$ duration = -omp_get_wtime()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    call libxsmm_imm(m, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1))
    do j = 1, k
        call libxsmm_imm(m, n, n, a(:,:,j,i), dy, tm2(:,:,j))
    enddo
    call libxsmm_imm(m*n, k, k, reshape(a(:,:,:,i), (/m*n,k/)), dz, tm3(:,:,1))
    !DEC$ vector aligned nontemporal
    c(:,:,:,i) = g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*tm3
  END DO
  !$OMP MASTER
  !$ duration = duration + omp_get_wtime()
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  call performance(duration, m, n, k, s)
  if (check /= 0) call validate(d, c)

  WRITE(*, "(A)") "Streamed... (mxm)"
  !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, g1, g2, g3, c, m, n, k, f1, f2, f3)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3=0
  !$OMP MASTER
  !$ duration = -omp_get_wtime()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    call mxmf2(dx, m, a(:,:,:,i), m, tm1, n*k)
    do j = 1, k
        call mxmf2(a(:,:,j,i), m, dy, n, tm2(:,:,j), n)
    enddo
    call mxmf2(a(:,:,:,i), m*n, dz, k, tm3, k)
    !DEC$ vector aligned nontemporal
    c(:,:,:,i) = g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*tm3
  END DO
  !$OMP MASTER
  !$ duration = duration + omp_get_wtime()
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  call performance(duration, m, n, k, s)
  if (check /= 0) call validate(d, c)

  WRITE(*, "(A)") "Streamed... (auto-dispatched)"
  !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, g1, g2, g3, c, m, n, k, f1, f2, f3)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3=0
  !$OMP MASTER
  !$ duration = -omp_get_wtime()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    call libxsmm_mm(m, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1))
    do j = 1, k
        call libxsmm_mm(m, n, n, a(:,:,j,i), dy, tm2(:,:,j))
    enddo
    call libxsmm_mm(m*n, k, k, reshape(a(:,:,:,i), (/m*n,k/)), dz, tm3(:,:,1))
    !DEC$ vector aligned nontemporal
    c(:,:,:,i) = g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*tm3
  END DO
  !$OMP MASTER
  !$ duration = duration + omp_get_wtime()
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  call performance(duration, m, n, k, s)
  if (check /= 0) call validate(d, c)

  WRITE(*, "(A)") "Streamed... (specialized)"
  !$OMP PARALLEL PRIVATE(i) !DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, g1, g2, g3, c, m, n, k, f1, f2, f3)
  ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
  tm1 = 0; tm2 = 0; tm3=0

  f1 = libxsmm_mm_dispatch(m, n*k, m, T)
  f2 = libxsmm_mm_dispatch(m, n, n, T)
  f3 = libxsmm_mm_dispatch(m*n, k, k, T)
  if (C_ASSOCIATED(f1)) then
    CALL C_F_PROCPOINTER(f1, dmm1)
  else
    write(*,*) "f1 not built"
  endif
  if (C_ASSOCIATED(f2)) then
    CALL C_F_PROCPOINTER(f2, dmm2)
  else
    write(*,*) "f2 not built"
  endif
  if (C_ASSOCIATED(f3)) then
    CALL C_F_PROCPOINTER(f3, dmm3)
  else
    write(*,*) "f3 not built"
  endif
  !$OMP MASTER
  !$ duration = -omp_get_wtime()
  !$OMP END MASTER
  !$OMP DO
  DO i = LBOUND(a, 4), UBOUND(a, 4)
    CALL dmm1(dx, a(1,1,1,i), tm1)
    do j = 1, k
        call dmm2(a(1,1,j,i), dy, tm2(1,1,j))
    enddo
    CALL dmm3(a(1,1,1,i), dz, tm3)
    !DEC$ vector aligned nontemporal
    c(:,:,:,i) = g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*tm3
  END DO
  !$OMP MASTER
  !$ duration = duration + omp_get_wtime()
  !$OMP END MASTER
  ! Deallocate thread-local arrays
  DEALLOCATE(tm1, tm2, tm3)
  !$OMP END PARALLEL

  call performance(duration, m, n, k, s)
  if (check /= 0) call validate(d, c)

  ! Deallocate global arrays
  DEALLOCATE(a)
  deallocate(g1, g2, g3)
  deallocate(dx, dy, dz)
  DEALLOCATE(c)

contains

subroutine validate(ref, test)
  real(T), dimension(:,:,:,:), intent(in) :: ref, test

  WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", MAXVAL((ref - test) * (ref - test))
end subroutine validate

subroutine performance(duration, m, n, k, s)
  real(8), intent(in) :: duration
  integer, intent(in) :: m, n, k
  integer(8), intent(in) :: s

  IF (0.LT.duration) THEN
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:", &
      (s * m * n * k * (2*(m+n+k) + 2) * 1D-9 / duration), " GFLOPS/s"
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ", &
      (s * m * n * k * (5) * T / (duration * LSHIFT(1_8, 30))), " GB/s"
  ENDIF
  WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ", 1D3 * duration, " ms"
end subroutine performance

END PROGRAM
