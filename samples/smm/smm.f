!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXSMM library.                             !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/libxsmm/libxsmm/              !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!
! Hans Pabst (Intel Corp.), Alexander Heinecke (Intel Corp.)
!=======================================================================!

      PROGRAM smm
        USE :: LIBXSMM, libxsmm_mmcall => libxsmm_dmmcall_abc
        !$ USE omp_lib
        IMPLICIT NONE

        INTEGER, PARAMETER :: T = KIND(0D0)

        REAL(T), ALLOCATABLE :: a(:,:,:), b(:,:,:), c(:,:,:)
        REAL(T), ALLOCATABLE :: sumxsmm(:,:), sumblas(:,:)
        REAL(T), ALLOCATABLE, SAVE :: sumtmp(:,:)
        !DIR$ ATTRIBUTES ALIGN:64 :: a, b, c, sumxsmm, sumtmp
        !$OMP THREADPRIVATE(sumtmp)
        TYPE(LIBXSMM_DMMFUNCTION) :: xmm
        INTEGER(8) :: i, r, s, size0, size1, size2, repetitions, start
        TYPE(LIBXSMM_MATDIFF_INFO) :: diff, max_diff
        INTEGER(LIBXSMM_BLASINT_KIND) :: m, n, k
        DOUBLE PRECISION :: duration, scale
        CHARACTER(32) :: argv
        INTEGER :: argc

        argc = COMMAND_ARGUMENT_COUNT()
        IF (1 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(1, argv)
          READ(argv, "(I32)") m
        ELSE
          m = 23
        END IF
        IF (3 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(3, argv)
          READ(argv, "(I32)") k
        ELSE
          k = m
        END IF
        IF (2 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(2, argv)
          READ(argv, "(I32)") n
        ELSE
          n = k
        END IF
        IF (4 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(4, argv)
          READ(argv, "(I32)") size1
        ELSE
          size1 = 0
        END IF
        IF (5 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(5, argv)
          READ(argv, "(I32)") size2
        ELSE
          size2 = 0 ! 1 repetition by default
        END IF

        ! Initialize LIBXSMM
        CALL libxsmm_init()

        ! JIT-compile stream(a,b)-kernel
        CALL libxsmm_dispatch(xmm, m, n, k,                             &
     &    alpha=REAL(1,T), beta=REAL(1,T))

        ! workload is about 2 GByte in memory by default
        size0 = (m * k + k * n + m * n) * T ! size of stream(a,b,c)-element in Byte
        size1 = MERGE(2048_8, MERGE(size1, ISHFT(ABS(size0 * size1)     &
     &            + ISHFT(1, 20) - 1, -20), 0.LE.size1), 0.EQ.size1)
        size2 = ISHFT(MERGE(MAX(size2, size1), ISHFT(ABS(size2) * size0 &
     &            + ISHFT(1, 20) - 1, -20), 0.LE.size2), 20) / size0
        s = ISHFT(size1, 20) / size0
        repetitions = size2 / s
        scale = 1D0 / s
        duration = 0

        CALL libxsmm_matdiff_clear(max_diff)
        ALLOCATE(a(m,k,s))
        ALLOCATE(b(k,n,s))

        ! Initialize a, b
        !$OMP PARALLEL DO PRIVATE(i) DEFAULT(NONE) SHARED(s, a, b, scale)
        DO i = 1, s
          CALL init(42, a(:,:,i), scale, i - 1)
          CALL init(24, b(:,:,i), scale, i - 1)
        END DO
        !$OMP END PARALLEL DO

        WRITE(*, "(3(A,I0),A,I0,A,I0,A,I0)")                            &
     &    "m=", m, " n=", n, " k=", k, " elements=", UBOUND(a, 3),      &
     &    " size=", size1, " MB repetitions=", repetitions

        ! compute reference solution and warmup BLAS library
        ALLOCATE(sumblas(m,n))
        sumblas(:,:) = REAL(0, T)
        !$OMP PARALLEL REDUCTION(+:sumblas) PRIVATE(i, r)               &
        !$OMP   DEFAULT(NONE) SHARED(m, n, k, a, b, repetitions)
        ALLOCATE(sumtmp(m,n))
        sumtmp(:,:) = REAL(0, T)
        DO r = 1, repetitions
          !$OMP DO
          DO i = LBOUND(a, 3), UBOUND(a, 3)
            ! PGI: cannot deduce generic procedure (libxsmm_blas_gemm)
            CALL libxsmm_blas_dgemm(m=m, n=n, k=k,                      &
     &              a=a(:,:,i), b=b(:,:,i), c=sumtmp)
          END DO
        END DO
        sumblas(:,:) = sumblas(:,:) + sumtmp(:UBOUND(sumblas,1),:)
        ! Deallocate thread-local arrays
        DEALLOCATE(sumtmp)
        !$OMP END PARALLEL

        WRITE(*, "(A)") "Streamed (A,B)... (BLAS)"
        ALLOCATE(sumxsmm(m,n))
        sumxsmm(:,:) = REAL(0, T)
        !$OMP PARALLEL REDUCTION(+:sumxsmm) PRIVATE(i, r, start)        &
        !$OMP   DEFAULT(NONE)                                           &
        !$OMP   SHARED(m, n, k, a, b, duration, repetitions)
        ALLOCATE(sumtmp(m,n))
        sumtmp(:,:) = REAL(0, T)
        !$OMP MASTER
        start = libxsmm_timer_tick()
        !$OMP END MASTER
        !$OMP BARRIER
        DO r = 1, repetitions
          !$OMP DO
          DO i = LBOUND(a, 3), UBOUND(a, 3)
            ! PGI: cannot deduce generic procedure (libxsmm_blas_gemm)
            CALL libxsmm_blas_dgemm(m=m, n=n, k=k,                      &
     &              a=a(:,:,i), b=b(:,:,i), c=sumtmp)
          END DO
        END DO
        !$OMP BARRIER
        !$OMP MASTER
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
        !$OMP END MASTER
        sumxsmm(:,:) = sumxsmm(:,:) + sumtmp(:UBOUND(sumxsmm,1),:)
        ! Deallocate thread-local arrays
        DEALLOCATE(sumtmp)
        !$OMP END PARALLEL
        CALL performance(duration, m, n, k, size2, m * k + k * n)

        WRITE(*, "(A)") "Streamed (A,B)... (auto-dispatched)"
        sumxsmm(:,:) = REAL(0, T)
        !$OMP PARALLEL REDUCTION(+:sumxsmm) PRIVATE(i, r, start)        &
        !$OMP   DEFAULT(NONE)                                           &
        !$OMP   SHARED(m, n, k, a, b, duration, repetitions)
        ALLOCATE(sumtmp(m,n))
        sumtmp(:,:) = REAL(0, T)
        !$OMP MASTER
        start = libxsmm_timer_tick()
        !$OMP END MASTER
        !$OMP BARRIER
        DO r = 1, repetitions
          !$OMP DO
          DO i = LBOUND(a, 3), UBOUND(a, 3)
            ! PGI: cannot deduce generic procedure (libxsmm_gemm)
            CALL libxsmm_dgemm(m=m, n=n, k=k,                           &
     &              a=a(:,:,i), b=b(:,:,i), c=sumtmp)
          END DO
        END DO
        !$OMP BARRIER
        !$OMP MASTER
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
        !$OMP END MASTER
        sumxsmm(:,:) = sumxsmm(:,:) + sumtmp(:UBOUND(sumxsmm,1),:)
        ! Deallocate thread-local arrays
        DEALLOCATE(sumtmp)
        !$OMP END PARALLEL
        CALL performance(duration, m, n, k, size2, m * k + k * n)
        CALL libxsmm_matdiff(diff, LIBXSMM_DATATYPE_F64, m, n,          &
     &    libxsmm_ptr(sumblas), libxsmm_ptr(sumxsmm))
        WRITE(*, "(1A,A,F10.1)") CHAR(9), "diff:      ", diff%l2_abs
        CALL libxsmm_matdiff_reduce(max_diff, diff)

        IF (libxsmm_available(xmm)) THEN
          sumxsmm(:,:) = REAL(0, T)
          WRITE(*, "(A)") "Streamed (A,B)... (specialized)"
          !$OMP PARALLEL REDUCTION(+:sumxsmm) PRIVATE(i, r, start)
            !DEFAULT(NONE) SHARED(m, n, a, b, duration, repetitions, xmm)
          ALLOCATE(sumtmp(m,n))
          sumtmp(:,:) = REAL(0, T)
          !$OMP MASTER
          start = libxsmm_timer_tick()
          !$OMP END MASTER
          !$OMP BARRIER
          DO r = 1, repetitions
            !$OMP DO
            DO i = LBOUND(a, 3), UBOUND(a, 3)
              CALL libxsmm_mmcall(xmm, a(:,:,i), b(:,:,i), sumtmp)
            END DO
          END DO
          !$OMP BARRIER
          !$OMP MASTER
          duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
          !$OMP END MASTER
          sumxsmm(:,:) = sumxsmm(:,:) + sumtmp(:UBOUND(sumxsmm,1),:)
          ! Deallocate thread-local arrays
          DEALLOCATE(sumtmp)
          !$OMP END PARALLEL
          CALL performance(duration, m, n, k, size2, m * k + k * n)
          CALL libxsmm_matdiff(diff, LIBXSMM_DATATYPE_F64, m, n,        &
     &      libxsmm_ptr(sumblas), libxsmm_ptr(sumxsmm))
          WRITE(*, "(1A,A,F10.1)") CHAR(9), "diff:      ", diff%l2_abs
          CALL libxsmm_matdiff_reduce(max_diff, diff)
        END IF

        WRITE(*, "(A)") "Streamed (A,B,C/Beta=0)... (specialized)"
        ALLOCATE(c(m,n,s))
        start = libxsmm_timer_tick()
        DO r = 1, repetitions
          !$OMP PARALLEL DEFAULT(NONE) SHARED(a, b, c)
          CALL smm_stream_abc_beta0(a, b, c)
          !$OMP END PARALLEL
        END DO
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
        CALL performance(duration, m, n, k, size2,                      &
     &    m * k + k * n + m * n * 2) ! 2: assume RFO

        WRITE(*, "(A)") "Streamed (A,B,C/Beta=1)... (specialized)"
        start = libxsmm_timer_tick()
        DO r = 1, repetitions
          !$OMP PARALLEL DEFAULT(NONE) SHARED(a, b, c)
          CALL smm_stream_abc_beta1(a, b, c)
          !$OMP END PARALLEL
        END DO
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
        CALL performance(duration, m, n, k, size2,                      &
     &    m * k + k * n + m * n * 2) ! 2: assume RFO

        ! Deallocate global arrays
        DEALLOCATE(a, b, c)
        DEALLOCATE(sumxsmm)
        DEALLOCATE(sumblas)

        ! finalize LIBXSMM
        CALL libxsmm_finalize()

        IF (1.LT.(max_diff%l2_rel)) STOP 1

      CONTAINS
        PURE SUBROUTINE init(seed, matrix, scale, n)
          INTEGER, INTENT(IN) :: seed
          REAL(T), INTENT(OUT) :: matrix(:,:)
          REAL(8), INTENT(IN) :: scale
          INTEGER(8), INTENT(IN), OPTIONAL :: n
          INTEGER(8) :: minval, addval, maxval
          INTEGER :: ld, i, j
          REAL(8) :: val, norm
          ld = UBOUND(matrix, 1) - LBOUND(matrix, 1) + 1
          minval = MERGE(n, 0_8, PRESENT(n)) + seed
          addval = (UBOUND(matrix, 1) - LBOUND(matrix, 1)) * ld         &
     &           + (UBOUND(matrix, 2) - LBOUND(matrix, 2))
          maxval = MAX(ABS(minval), addval)
          norm = MERGE(scale / maxval, scale, 0.NE.maxval)
          DO j = LBOUND(matrix, 2), UBOUND(matrix, 2)
            DO i = LBOUND(matrix, 1),                                   &
     &             LBOUND(matrix, 1) + UBOUND(matrix, 1) - 1
              val = (i - LBOUND(matrix, 1)) * ld                        &
     &            + (j - LBOUND(matrix, 2)) + minval
              matrix(i,j) = norm * (val - 0.5D0 * addval)
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

        SUBROUTINE performance(duration, m, n, k, s, c)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k, c
          INTEGER(8), INTENT(IN) :: s
          REAL(T), INTENT(IN) :: duration
          IF (0.LT.duration) THEN
            WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "performance:",         &
     &        2D0 * s * m * n * k * 1D-9 / duration, " GFLOPS/s"
            WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ",         &
     &        s * c * T / (duration * ISHFT(1_8, 30)), " GB/s"
          END IF
          WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ",           &
     &        1D3 * duration, " ms"
        END SUBROUTINE

        SUBROUTINE smm_stream_abc_beta0(a, b, c)
          REAL(T), INTENT(IN)  :: a(:,:,:), b(:,:,:)
          REAL(T), INTENT(OUT) :: c(:,:,:)
          TYPE(LIBXSMM_DMMFUNCTION) :: xmm
          INTEGER(8) :: i
          CALL libxsmm_dispatch(xmm, SIZE(c,1), SIZE(c,2), SIZE(b,1),   &
     &      flags=MERGE(                                                &
     &        LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0,                &
     &        LIBXSMM_GEMM_FLAG_BETA_0,                                 &
     &        libxsmm_aligned(libxsmm_ptr(c(1,1,1)),                    &
     &        SIZE(c,1) * SIZE(c,2) * T)))
          !$OMP DO
          DO i = LBOUND(c, 3, 8), UBOUND(c, 3, 8)
            CALL libxsmm_mmcall(xmm, a(:,:,i), b(:,:,i), c(:,:,i))
          END DO
        END SUBROUTINE

        SUBROUTINE smm_stream_abc_beta1(a, b, c)
          REAL(T), INTENT(IN)    :: a(:,:,:), b(:,:,:)
          REAL(T), INTENT(INOUT) :: c(:,:,:)
          TYPE(LIBXSMM_DMMFUNCTION) :: xmm
          INTEGER(8) :: i
          CALL libxsmm_dispatch(xmm, SIZE(c,1), SIZE(c,2), SIZE(b,1))
          !$OMP DO
          DO i = LBOUND(c, 3, 8), UBOUND(c, 3, 8)
            CALL libxsmm_mmcall(xmm, a(:,:,i), b(:,:,i), c(:,:,i))
          END DO
        END SUBROUTINE
      END PROGRAM

