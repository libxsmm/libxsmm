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

#define XSMM_DIRECT
!#define XSMM_DISPATCH

PROGRAM stpm
#ifdef XSMM_DIRECT
  use libxsmm, only : LIBXSMM_DOUBLE_PRECISION, LIBXSMM_ALIGNED_MAX, LIBXSMM_DMM_FUNCTION
#else
  USE :: LIBXSMM
#endif

  !$ USE omp_lib
  IMPLICIT NONE

  INTEGER, PARAMETER :: T = LIBXSMM_DOUBLE_PRECISION
  INTEGER, PARAMETER :: MAX_NTHREADS = 512

  TYPE :: libxsmm_matrix
    REAL(T), POINTER :: matrix(:,:)
  END TYPE libxsmm_matrix

  REAL(T), allocatable, target :: a(:,:,:,:), c(:,:,:,:)
  real(T), allocatable, target :: dx(:,:), dy(:,:), dz(:,:)
  REAL(T), ALLOCATABLE, TARGET :: d(:,:,:,:)
  REAL(T), ALLOCATABLE, TARGET, SAVE :: tmp(:,:,:), tm1(:,:,:), tm2(:,:,:), tm3(:,:,:)
  !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNED_MAX :: a, c, tmp
  !$OMP THREADPRIVATE(tmp, tm1, tm2, tm3)
!  PROCEDURE(LIBXSMM_XMM_FUNCTION), POINTER :: xmm     ! Fully ploymorph variant
  PROCEDURE(LIBXSMM_DMM_FUNCTION), POINTER :: dmm1, dmm2, dmm3, dmm
  INTEGER :: argc, m, n, k, ld, routine, check
  INTEGER(8) :: i, j, s, ix, iy, iz
  CHARACTER(32) :: argv
  TYPE(C_FUNPTR) :: f1, f2, f3, f

  REAL(8) :: duration

#ifdef XSMM_DIRECT
  interface libxsmm_dmm_8_8_8
    subroutine libxsmm_dmm_8_8_8(a,b,c) BIND(C)
      use iso_c_binding, only : c_ptr
      type(c_ptr), value :: a, b, c
    end subroutine
  end interface
  interface libxsmm_dmm_64_8_8
    subroutine libxsmm_dmm_64_8_8(a,b,c) BIND(C)
      use iso_c_binding, only : c_ptr
      type(c_ptr), value :: a, b, c
    end subroutine
  end interface
  interface libxsmm_dmm_8_64_8
    subroutine libxsmm_dmm_8_64_8(a,b,c) BIND(C)
      use iso_c_binding, only : c_ptr
      type(c_ptr), value :: a, b, c
    end subroutine
  end interface
#endif


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
  s = LSHIFT(INT8(MAX(i, 0)), 30) / ((m * k + k * n + m * n) * T)

  ALLOCATE(c(m,n,k,s))
  ALLOCATE(a(m,n,k,s))
  ALLOCATE(dx(m,m), dy(n,n), dz(k,k))

  ! Initialize a, b
  !@TODO figure out how to allocate a,b matrices cont. without
  ! additional copies when call LIBXSMM in the F90 compiler
  !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(a, m, n, k, s)
  DO i = 1, s
    do ix = 1, m
      do iy = 1, n
        do iz = 1, k
          a(ix,iy,iz,i) = ix + iy*m + iz*m*n
        enddo
      enddo
    enddo
  END DO 
  dx = 1.; dy = 1.; dz = 1.
  c = 0

  WRITE (*, "(A,I3,A,I3,A,I3,A,I10)") "m=", m, " n=", n, " k=", k, " size=", UBOUND(a, 4) 

#if 0
  CALL GETENV("CHECK", argv)
  READ(argv, "(I32)") check
  IF (0.NE.check) THEN
    ALLOCATE(d(m,n))
    d(:,:) = 0
    !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) SHARED(duration, a, b, d, m, n, k)
    ALLOCATE(tmp(libxsmm_align_value(libxsmm_ld(m,n),T,LIBXSMM_ALIGNED_STORES),libxsmm_ld(n,m)))
    tmp(:,:) = 0
    !$OMP DO
    DO i = LBOUND(a, 1), UBOUND(a, 1)
      CALL libxsmm_blasmm(m, n, k, a(i)%matrix, b(i)%matrix, tmp)
    END DO
    !$OMP CRITICAL
    d(:,:) = d(:,:) + tmp(:UBOUND(d,1),:)
    !$OMP END CRITICAL
    ! Deallocate thread-local arrays
    DEALLOCATE(tmp)
    !$OMP END PARALLEL
  END IF
#endif

  IF (0.GT.routine) THEN
    WRITE(*, "(A)") "Streamed... (auto-dispatched)"
    !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) SHARED(duration, a, dx, dy, dz, c, m, n, k, f1, f2, f3)
    ALLOCATE(tm1(m,n,k), tm2(m,n,k), tm3(m,n,k))
    tm1 = 0; tm2 = 0; tm3=0
#ifdef XSMM_DIRECT
    !$OMP MASTER
    !$ duration = -omp_get_wtime()
    !$OMP END MASTER
    !$OMP DO
    DO i = LBOUND(a, 4), UBOUND(a, 4)
      call libxsmm_dmm_8_64_8(c_loc(dx), c_loc(a(1,1,1,i)), c_loc(tm1(1,1,1)))
      do j = 1, k
        call libxsmm_dmm_8_8_8(c_loc(a(1,1,j,i)), c_loc(dy), c_loc(tm2(1,1,j)))
      enddo
      call libxsmm_dmm_64_8_8(c_loc(a(1,1,1,i)), c_loc(dz), c_loc(tm3(1,1,1)))
      c(:,:,:,i) = c(:,:,:,i) + tm1 + tm2 + tm3
    END DO
#else 
#ifdef XSMM_DISPATCH

    f1 = libxsmm_mm_dispatch(m, n*k, m, T)
    f2 = libxsmm_mm_dispatch(m, n, n, T)
    f3 = libxsmm_mm_dispatch(m*n, n, n, T)
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
      c(:,:,:,i) = c(:,:,:,i) + tm1 + tm2 + tm3
    END DO
#else
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
      c(:,:,:,i) = c(:,:,:,i) + tm1 + tm2 + tm3
    END DO
#endif
#endif
    !$OMP MASTER
    !$ duration = duration + omp_get_wtime()
    !$OMP END MASTER
    !$OMP CRITICAL
    !$OMP END CRITICAL
    ! Deallocate thread-local arrays
    DEALLOCATE(tm1, tm2, tm3)
    !$OMP END PARALLEL
  ELSE
#if 0
    f = MERGE(libxsmm_mm_dispatch(m, n, k, T), C_NULL_FUNPTR, 0.EQ.routine)
    IF (C_ASSOCIATED(f)) THEN
!      CALL C_F_PROCPOINTER(f, xmm)     ! Fully polymorph variant
      CALL C_F_PROCPOINTER(f, dmm)
      WRITE(*, "(A)") "Streamed... (specialized)"
      !$OMP PARALLEL PRIVATE(i) !DEFAULT(NONE) SHARED(duration, a, b, c, m, n, dmm)
      ALLOCATE(tmp(libxsmm_align_value(libxsmm_ld(m,n),T,LIBXSMM_ALIGNED_STORES),libxsmm_ld(n,m)))
      tmp(:,:) = 0
      !$OMP MASTER
      !$ duration = -omp_get_wtime()
      !$OMP END MASTER
      !$OMP DO
      DO i = LBOUND(a, 1), UBOUND(a, 1)
!        This in case of a polymorph call, we need C_LOC :-(, however the next line
!        violates Fortran standard :-(
!        CALL xmm(C_LOC(a(i)%matrix(:,:)), C_LOC(b(i)%matrix(:,:)), C_LOC(tmp))
        CALL dmm(a(i)%matrix, b(i)%matrix, tmp)
      END DO
      !$OMP MASTER
      !$ duration = duration + omp_get_wtime()
      !$OMP END MASTER
      !$OMP CRITICAL
      c(:,:) = c(:,:) + tmp(:UBOUND(c,1),:)
      !$OMP END CRITICAL
      ! Deallocate thread-local arrays
      DEALLOCATE(tmp)
      !$OMP END PARALLEL
    ELSE
      IF (0.EQ.routine) THEN
        WRITE(*, "(A)") "Streamed... (optimized; no specialization found)"
      ELSE
        WRITE(*, "(A)") "Streamed... (optimized)"
      ENDIF
      !$OMP PARALLEL PRIVATE(i) DEFAULT(NONE) SHARED(duration, a, b, c, m, n, k)
      ALLOCATE(tmp(libxsmm_align_value(libxsmm_ld(m,n),T,LIBXSMM_ALIGNED_STORES),libxsmm_ld(n,m)))
      tmp(:,:) = 0
      !$OMP MASTER
      !$ duration = -omp_get_wtime()
      !$OMP END MASTER
      !$OMP DO
      DO i = LBOUND(a, 1), UBOUND(a, 1)
        CALL libxsmm_imm(m, n, k, a(i)%matrix, b(i)%matrix, tmp)
      END DO
      !$OMP MASTER
      !$ duration = duration + omp_get_wtime()
      !$OMP END MASTER
      !$OMP CRITICAL
      c(:,:) = c(:,:) + tmp(:UBOUND(c,1),:)
      !$OMP END CRITICAL
      ! Deallocate thread-local arrays
      DEALLOCATE(tmp)
      !$OMP END PARALLEL
    ENDIF
#endif
  END IF

  IF (0.LT.duration) THEN
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:", &
      (2D0 * s * m * n * k * (m+n+k + .5) * 1D-9 / duration), " GFLOPS/s"
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ", &
      (s * (2 * m * n * k) * T / (duration * LSHIFT(1_8, 30))), " GB/s"
  ENDIF
  WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ", 1D3 * duration, " ms"
#if 0
  IF (0.NE.check) THEN
    WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", MAXVAL((c(:,:) - d(:,:)) * (c(:,:) - d(:,:)))
    DEALLOCATE(d)
  END IF
#endif

  ! Deallocate global arrays
  DEALLOCATE(a)
  deallocate(dx, dy, dz)
  DEALLOCATE(c)

CONTAINS
#if 0
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

  PURE SUBROUTINE init(seed, matrix, n)
    INTEGER, INTENT(IN) :: seed
    REAL(T), INTENT(OUT) :: matrix(:,:,:)
    INTEGER(8), INTENT(IN), OPTIONAL :: n
    INTEGER(8) :: minval, addval, maxval
    INTEGER :: ld, i, j, k
    REAL(8) :: value, norm
    ld = UBOUND(matrix, 1) - LBOUND(matrix, 1) + 1
    minval = MERGE(n, 0_8, PRESENT(n)) + seed
    addval = (UBOUND(matrix, 1) - LBOUND(matrix, 1)) * ld * ld + (UBOUND(matrix, 2) - LBOUND(matrix, 2)) * ld + (UBOUND(matrix, 3) - LBOUND(matrix, 3))
    maxval = MAX(ABS(minval), addval)
    norm = MERGE(1D0 / maxval, 1D0, 0.NE.maxval)
    do k = LBOUND(matrix, 3), UBOUND(matrix, 3)
      DO j = LBOUND(matrix, 2), UBOUND(matrix, 2)
        DO i = LBOUND(matrix, 1), LBOUND(matrix, 1) + UBOUND(matrix, 1) - 1
          value = (i - LBOUND(matrix, 1)) * ld *ld + (j - LBOUND(matrix, 2))*ld + (k - LBOUND(matrix,3)) + minval
          matrix(i,j) = norm * (value - 0.5D0 * addval)
        END DO
      END DO
    enddo
  END SUBROUTINE
#endif


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
END PROGRAM
