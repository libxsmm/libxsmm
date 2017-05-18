!*****************************************************************************!
!* Copyright (c) 2015-2017, Intel Corporation                                *!
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

        INTEGER, PARAMETER :: T = KIND(0D0)

        REAL(T), ALLOCATABLE, TARGET :: a(:,:,:), b(:,:,:)
        REAL(T), ALLOCATABLE, TARGET :: c(:,:), d(:,:)
        REAL(T), ALLOCATABLE, TARGET, SAVE :: tmp(:,:)
        !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNMENT :: a, b, c, tmp
        !$OMP THREADPRIVATE(tmp)
        TYPE(LIBXSMM_DMMFUNCTION) :: xmm
        DOUBLE PRECISION :: duration, scale, max_diff, diff
        INTEGER(8) :: i, r, s, size0, size1, size, repetitions, start
        INTEGER :: argc, m, n, k
        CHARACTER(32) :: argv

        argc = COMMAND_ARGUMENT_COUNT()
        IF (1 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(1, argv)
          READ(argv, "(I32)") m
        ELSE
          m = 23
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

        ! Eventually JIT-compile the requested kernel
        CALL libxsmm_dispatch(xmm, m, n, k)

        ! workload is about 2 GByte in memory by default
        size0 = (m * k + k * n) * T ! size of a single stream element in Byte
        size1 = MERGE(2048_8, MERGE(size1, ISHFT(ABS(size0 * size1)     &
     &            + ISHFT(1, 20) - 1, -20), 0.LE.size1), 0.EQ.size1)
        size = ISHFT(MERGE(MAX(size, size1), ISHFT(ABS(size) * size0    &
     &            + ISHFT(1, 20) - 1, -20), 0.LE.size), 20) / size0
        s = ISHFT(size1, 20) / size0
        repetitions = size / s
        scale = 1D0 / s
        duration = 0
        max_diff = 0

        ALLOCATE(c(m,n))
        ALLOCATE(a(m,k,s))
        ALLOCATE(b(k,n,s))

        ! Initialize a, b
        !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(s, a, b, scale)
        DO i = 1, s
          CALL init(42, a(:,:,i), scale, i - 1)
          CALL init(24, b(:,:,i), scale, i - 1)
        END DO

        WRITE(*, "(3(A,I0),A,I0,A,I0,A,I0)")                            &
     &    "m=", m, " n=", n, " k=", k, " elements=", UBOUND(a, 3),      &
     &    " size=", size1, "MB repetitions=", repetitions

        ! compute reference solution and warmup BLAS library
        ALLOCATE(d(m,n))
        d(:,:) = 0
        !$OMP PARALLEL REDUCTION(+:d) PRIVATE(i, r) &
        !$OMP   DEFAULT(NONE) SHARED(m, n, k, a, b, repetitions)
        ALLOCATE(tmp(m,n))
        tmp(:,:) = 0
        DO r = 1, repetitions
          !$OMP DO
          DO i = LBOUND(a, 3), UBOUND(a, 3)
            CALL libxsmm_blas_gemm(m=m, n=n, k=k,                       &
     &              a=a(:,:,i), b=b(:,:,i), c=tmp)
          END DO
        END DO
        d(:,:) = d(:,:) + tmp(:UBOUND(d,1),:)
        ! Deallocate thread-local arrays
        DEALLOCATE(tmp)
        !$OMP END PARALLEL

        WRITE(*, "(A)") "Streamed... (BLAS)"
        c(:,:) = 0
        !$OMP PARALLEL REDUCTION(+:c) PRIVATE(i, r, start) &
        !$OMP   DEFAULT(NONE) SHARED(m, n, k, a, b, duration, repetitions)
        ALLOCATE(tmp(m,n))
        tmp(:,:) = 0
        !$OMP MASTER
        start = libxsmm_timer_tick()
        !$OMP END MASTER
        !$OMP BARRIER
        DO r = 1, repetitions
          !$OMP DO
          DO i = LBOUND(a, 3), UBOUND(a, 3)
            CALL libxsmm_blas_gemm(m=m, n=n, k=k,                       &
     &              a=a(:,:,i), b=b(:,:,i), c=tmp)
          END DO
        END DO
        !$OMP BARRIER
        !$OMP MASTER
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
        !$OMP END MASTER
        c(:,:) = c(:,:) + tmp(:UBOUND(c,1),:)
        ! Deallocate thread-local arrays
        DEALLOCATE(tmp)
        !$OMP END PARALLEL
        CALL performance(duration, m, n, k, size)

        WRITE(*, "(A)") "Streamed... (auto-dispatched)"
        c(:,:) = 0
        !$OMP PARALLEL REDUCTION(+:c) PRIVATE(i, r, start) &
        !$OMP   DEFAULT(NONE) SHARED(m, n, k, a, b, duration, repetitions)
        ALLOCATE(tmp(m,n))
        tmp(:,:) = 0
        !$OMP MASTER
        start = libxsmm_timer_tick()
        !$OMP END MASTER
        !$OMP BARRIER
        DO r = 1, repetitions
          !$OMP DO
          DO i = LBOUND(a, 3), UBOUND(a, 3)
            CALL libxsmm_gemm(m=m, n=n, k=k,                            &
     &              a=a(:,:,i), b=b(:,:,i), c=tmp)
          END DO
        END DO
        !$OMP BARRIER
        !$OMP MASTER
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
        !$OMP END MASTER
        c(:,:) = c(:,:) + tmp(:UBOUND(c,1),:)
        ! Deallocate thread-local arrays
        DEALLOCATE(tmp)
        !$OMP END PARALLEL
        CALL performance(duration, m, n, k, size)
        diff = MAXVAL((c(:,:) - d(:,:)) * (c(:,:) - d(:,:)))
        WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", diff
        max_diff = MAX(diff, max_diff)

        IF (libxsmm_available(xmm)) THEN
          c(:,:) = 0
          WRITE(*, "(A)") "Streamed... (specialized)"
          !$OMP PARALLEL REDUCTION(+:c) PRIVATE(i, r, start)
            !DEFAULT(NONE) SHARED(m, n, a, b, duration, repetitions, xmm)
          ALLOCATE(tmp(m,n))
          tmp(:,:) = 0
          !$OMP MASTER
          start = libxsmm_timer_tick()
          !$OMP END MASTER
          !$OMP BARRIER
          DO r = 1, repetitions
            !$OMP DO
            DO i = LBOUND(a, 3), UBOUND(a, 3)
              CALL libxsmm_dmmcall(xmm, a(:,:,i), b(:,:,i), tmp)
            END DO
          END DO
          !$OMP BARRIER
          !$OMP MASTER
          duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
          !$OMP END MASTER
          c(:,:) = c(:,:) + tmp(:UBOUND(c,1),:)
          ! Deallocate thread-local arrays
          DEALLOCATE(tmp)
          !$OMP END PARALLEL
          CALL performance(duration, m, n, k, size)
          diff = MAXVAL((c(:,:) - d(:,:)) * (c(:,:) - d(:,:)))
          WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "diff:       ", diff
          max_diff = MAX(diff, max_diff)
        END IF

        ! Deallocate global arrays
        DEALLOCATE(a)
        DEALLOCATE(b)
        DEALLOCATE(c)
        DEALLOCATE(d)

        ! finalize LIBXSMM
        CALL libxsmm_finalize()

        IF (1.LT.max_diff) STOP 1

      CONTAINS
        PURE SUBROUTINE init(seed, matrix, scale, n)
          INTEGER, INTENT(IN) :: seed
          REAL(T), INTENT(OUT) :: matrix(:,:)
          REAL(8), INTENT(IN) :: scale
          INTEGER(8), INTENT(IN), OPTIONAL :: n
          INTEGER(8) :: minval, addval, maxval
          INTEGER :: ld, i, j
          REAL(8) :: value, norm
          ld = UBOUND(matrix, 1) - LBOUND(matrix, 1) + 1
          minval = MERGE(n, 0_8, PRESENT(n)) + seed
          addval = (UBOUND(matrix, 1) - LBOUND(matrix, 1)) * ld         &
     &           + (UBOUND(matrix, 2) - LBOUND(matrix, 2))
          maxval = MAX(ABS(minval), addval)
          norm = MERGE(scale / maxval, scale, 0.NE.maxval)
          DO j = LBOUND(matrix, 2), UBOUND(matrix, 2)
            DO i = LBOUND(matrix, 1),                                   &
     &             LBOUND(matrix, 1) + UBOUND(matrix, 1) - 1
              value = (i - LBOUND(matrix, 1)) * ld                      &
     &              + (j - LBOUND(matrix, 2)) + minval
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
          END IF
          i0 = LBOUND(matrix, 1)
          i1 = MIN(                                                     &
     &          MERGE(i0 + ld - 1, UBOUND(matrix, 1), PRESENT(ld)),     &
     &          UBOUND(matrix, 1))
          DO i = i0, i1
            DO j = LBOUND(matrix, 2), UBOUND(matrix, 2)
              WRITE(*, fmt, advance='NO') matrix(i,j)
            END DO
            WRITE(*, *)
          END DO
        END SUBROUTINE

        SUBROUTINE performance(duration, m, n, k, size)
          REAL(T), INTENT(IN) :: duration
          INTEGER, INTENT(IN) :: m, n, k
          INTEGER(8), INTENT(IN) :: size
          IF (0.LT.duration) THEN
            WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:",         &
     &        2D0 * size * m * n * k * 1D-9 / duration, " GFLOPS/s"
            WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ",         &
     &        size * (m * k + k * n) * T / (duration * ISHFT(1_8, 30)), &
     &        " GB/s"
          END IF
          WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ",           &
     &        1D3 * duration, " ms"
        END SUBROUTINE
      END PROGRAM
