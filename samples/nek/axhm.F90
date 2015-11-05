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

  INTEGER, PARAMETER :: T = LIBXSMM_DOUBLE_PRECISION
  REAL(T), PARAMETER :: alpha = 1, beta = 1

  REAL(T), allocatable, dimension(:,:,:,:), target :: a, c, g1, g2, g3, b, d
  real(T), allocatable, target :: dx(:,:), dy(:,:), dz(:,:)
  REAL(T), ALLOCATABLE, TARGET, SAVE :: tm1(:,:,:), tm2(:,:,:), tm3(:,:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNED_MAX :: a, c, g1, g2, g3, d
  !$OMP THREADPRIVATE(tm1, tm2, tm3)
  PROCEDURE(LIBXSMM_DMM_FUNCTION), POINTER :: dmm1, dmm2, dmm3
  INTEGER :: argc, m, n, k, routine, check
  INTEGER(8) :: i, j, s, ix, iy, iz, start
  CHARACTER(32) :: argv
  TYPE(C_FUNPTR) :: f1, f2, f3

  REAL(8) :: duration, h1, h2

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
    READ(argv, "(I32)") routine
  ELSE
    routine = -1
  END IF
  IF (5 <= argc) THEN
    CALL GETARG(5, argv)
    READ(argv, "(I32)") i
  ELSE
    i = 2 ! 2 GByte for A and B (and C, but this currently not used by the F90 test)
  END IF
  s = LSHIFT(INT8(MAX(i, 0)), 30) / ((m * n * k) * T * 6)

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

  WRITE (*, "(A,I3,A,I3,A,I3,A,I10)") "m=", m, " n=", n, " k=", k, " size=", UBOUND(a, 4) 

  IF (0.GT.routine) THEN
    WRITE(*, "(A)") "Streamed... (auto-dispatched)"
    !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) SHARED(duration, a, b, dx, dy, dz, g1, g2, g3, c, m, n, k, f1, f2, f3, h1, h2)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
    tm1 = 0; tm2 = 0; tm3=0
    !$OMP MASTER
    start = libxsmm_timer_tick()
    !$OMP END MASTER
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL libxsmm_mm(alpha, beta, m, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1))
      do j = 1, k
          CALL libxsmm_mm(alpha, beta, m, n, n, a(:,:,j,i), dy, tm2(:,:,j))
      enddo
      CALL libxsmm_mm(alpha, beta, m*n, k, k, reshape(a(:,:,:,i), (/m*n,k/)), dz, tm3(:,:,1))
      CALL updateC( c(:,:,:,i), g1(:,:,:,i), tm1, g2(:,:,:,i), tm2, &
                    g3(:,:,:,i), tm3, b(:,:,:,i), a(:,:,:,i), h1, h2 ) 
    END DO
    !$OMP MASTER
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
    !$OMP END MASTER
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL
  else if (0 .eq. routine) then
    WRITE(*, "(A)") "Streamed... (compiled)"
    !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, g1, g2, g3, b, c, m, n, k, f1, f2, f3, h1, h2)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
    tm1 = 0; tm2 = 0; tm3=0
    !$OMP MASTER
    start = libxsmm_timer_tick()
    !$OMP END MASTER
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL libxsmm_imm(alpha, beta, m, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1))
      do j = 1, k
          CALL libxsmm_imm(alpha, beta, m, n, n, a(:,:,j,i), dy, tm2(:,:,j))
      enddo
      CALL libxsmm_imm(alpha, beta, m*n, k, k, reshape(a(:,:,:,i), (/m*n,k/)), dz, tm3(:,:,1))
      CALL updateC( c(:,:,:,i), g1(:,:,:,i), tm1, g2(:,:,:,i), tm2, &
                    g3(:,:,:,i), tm3, b(:,:,:,i), a(:,:,:,i), h1, h2 ) 
    END DO
    !$OMP MASTER
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
    !$OMP END MASTER
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL
  ELSE IF (routine == 100) then
    WRITE(*, "(A)") "Streamed... (mxm)"
    !$OMP PARALLEL PRIVATE(i, start) DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, g1, g2, g3, b, c, m, n, k, f1, f2, f3, h1, h2)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
    tm1 = 0; tm2 = 0; tm3=0
    !$OMP MASTER
    start = libxsmm_timer_tick()
    !$OMP END MASTER
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL mxmf2(alpha, beta, dx, m, a(:,:,:,i), m, tm1, n*k)
      do j = 1, k
          CALL mxmf2(alpha, beta, a(:,:,j,i), m, dy, n, tm2(:,:,j), n)
      enddo
      CALL mxmf2(alpha, beta, a(:,:,:,i), m*n, dz, k, tm3, k)
      CALL updateC( c(:,:,:,i), g1(:,:,:,i), tm1, g2(:,:,:,i), tm2, &
                    g3(:,:,:,i), tm3, b(:,:,:,i), a(:,:,:,i), h1, h2 ) 
    END DO
    !$OMP MASTER
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
    !$OMP END MASTER
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL
  ELSE
    WRITE(*, "(A)") "Streamed... (specialized)"
    !$OMP PARALLEL PRIVATE(i, start) !DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, g1, g2, g3, b, c, m, n, k, f1, f2, f3, h1, h2)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
    tm1 = 0; tm2 = 0; tm3=0

    f1 = libxsmm_dispatch(alpha, beta, m, n*k, m, T)
    f2 = libxsmm_dispatch(alpha, beta, m, n, n, T)
    f3 = libxsmm_dispatch(alpha, beta, m*n, k, k, T)
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
    start = libxsmm_timer_tick()
    !$OMP END MASTER
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      CALL dmm1(alpha, beta, dx, a(1,1,1,i), tm1)
      do j = 1, k
          CALL dmm2(alpha, beta, a(1,1,j,i), dy, tm2(1,1,j))
      enddo
      CALL dmm3(alpha, beta, a(1,1,1,i), dz, tm3)
      CALL stream_update_axhm( g1(1,1,1,i), g2(1,1,1,i), g3(1,1,1,i), &
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
  END IF

  CALL GETENV("CHECK", argv)
  READ(argv, "(I32)") check
  IF (0.NE.check) THEN
    WRITE(*, "(A)") "Calculating check..."
    ALLOCATE(d(m,n,k,s))
    d = 0

    !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) SHARED(duration, a, b, dx, dy, dz, g1, g2, g3, d, m, n, k, f1, f2, f3, h1, h2)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m*n,k,1))
    tm1 = 0; tm2 = 0; tm3=0
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      call libxsmm_blasmm(alpha, beta, m, n*k, m, dx, reshape(a(:,:,:,i), (/m,n*k/)), tm1(:,:,1))
      do j = 1, k
          call libxsmm_blasmm(alpha, beta, m, n, n, a(:,:,j,i), dy, tm2(:,:,j))
      enddo
      call libxsmm_blasmm(alpha, beta, m*n, k, k, reshape(a(:,:,:,i), (/m*n,k/)), dz, tm3(:,:,1))
      !DEC$ vector aligned nontemporal
      d(:,:,:,i) = h1*(g1(:,:,:,i)*tm1 + g2(:,:,:,i)*tm2 + g3(:,:,:,i)*reshape(tm3, (/m,n,k/))) &
                 + h2*b(:,:,:,i)*a(:,:,:,i)
    END DO
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL
  END IF

  IF (0.LT.duration) THEN
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:", &
      (s * m * n * k * (2*(m+n+k) + 2 + 4) * 1D-9 / duration), " GFLOPS/s"
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ", &
      (s * m * n * k * (6) * T / (duration * LSHIFT(1_8, 30))), " GB/s"
  ENDIF
  WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ", 1D3 * duration, " ms"
  IF (0.NE.check) THEN
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", MAXVAL((c - d) * (c - d))
    DEALLOCATE(d)
  END IF

  ! Deallocate global arrays
  DEALLOCATE(a)
  DEALLOCATE(b)
  deallocate(g1, g2, g3)
  deallocate(dx, dy, dz)
  DEALLOCATE(c)

CONTAINS
  SUBROUTINE updateC( c, g1, tm1, g2, tm2, g3, tm3, b, a, h1, h2 )
    REAL(T), INTENT(INOUT) :: c(:,:,:)
    REAL(T), INTENT(IN)    :: b(:,:,:), a(:,:,:)
    REAL(T), INTENT(IN)    :: g1(:,:,:), tm1(:,:,:), g2(:,:,:), tm2(:,:,:), g3(:,:,:), tm3(:,:,:)
    REAL(T), INTENT(IN)    :: h1, h2
!    TYPE(C_PTR)            :: ptr
!    INTEGER(C_INTPTR_T)    :: addr
!    ptr = C_LOC(c(1,1,1))
!    addr = TRANSFER(ptr, C_INTPTR_T) 
!    WRITE (*,*) MOD(addr, 64)

    !DEC$ vector nontemporal
    c = h1*(g1*tm1 + g2*tm2 + g3*tm3) + h2*(b*a)
  END SUBROUTINE
END PROGRAM
