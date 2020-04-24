!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXSMM library.                             !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxsmm/                  !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!
! Hans Pabst (Intel Corp.)
!=======================================================================!

      PROGRAM matcopy
        USE :: LIBXSMM, ONLY: LIBXSMM_BLASINT_KIND,                     &
     &                        libxsmm_timer_duration,                   &
     &                        libxsmm_timer_tick,                       &
     &                        libxsmm_matcopy_omp,                      &
     &                        libxsmm_matcopy,                          &
     &                        libxsmm_init,                             &
     &                        ptr => libxsmm_ptr
        IMPLICIT NONE

        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER, PARAMETER :: S = T
        INTEGER, PARAMETER :: W = 50
        REAL(T), PARAMETER :: X = REAL(-1, T)

        REAL(T), ALLOCATABLE, TARGET :: a1(:), b1(:)
        !DIR$ ATTRIBUTES ALIGN:64 :: a1, b1
        INTEGER(LIBXSMM_BLASINT_KIND) :: m, n, ldi, ldo, h, i, j
        REAL(T), POINTER :: an(:,:,:), bn(:,:,:)
        DOUBLE PRECISION :: d, duration(4)
        INTEGER(8) :: start
        INTEGER :: r, nrepeat
        INTEGER :: k, nmb
        INTEGER :: nbytes

        INTEGER :: argc, check, zero
        CHARACTER(32) :: argv

        ! CHECK: 0 (OFF), 1 (ON)
        CALL GET_ENVIRONMENT_VARIABLE("CHECK", argv, check)
        IF (0.EQ.check) THEN ! check length
          check = 1 ! default state
        ELSE ! read given value
          READ(argv, "(I32)") check
        END IF
        ! ZERO: 0 (OFF), 1 (ZERO), 2 (COPY+ZERO)
        CALL GET_ENVIRONMENT_VARIABLE("ZERO", argv, zero)
        IF (0.EQ.zero) THEN ! check length
          zero = 0 ! default state
        ELSE ! read given value
          READ(argv, "(I32)") zero
        END IF

        argc = COMMAND_ARGUMENT_COUNT()
        IF (1 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(1, argv)
          READ(argv, "(I32)") m
        ELSE
          m = 4096
        END IF
        IF (2 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(2, argv)
          READ(argv, "(I32)") n
        ELSE
          n = m
        END IF
        IF (3 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(3, argv)
          READ(argv, "(I32)") ldi
          ldi = MAX(ldi, m)
        ELSE
          ldi = m
        END IF
        IF (4 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(4, argv)
          READ(argv, "(I32)") ldo
          ldo = MAX(ldi, m)
        ELSE
          ldo = ldi
        END IF
        IF (5 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(5, argv)
          READ(argv, "(I32)") nrepeat
        ELSE
          nrepeat = 5
        END IF
        IF (6 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(6, argv)
          READ(argv, "(I32)") nmb
          IF (0.GE.nmb) nmb = 2048
        ELSE ! 2 GB by default
          nmb = 2048
        END IF

        nbytes = m * n * S ! size in Byte
        k = INT(ISHFT(INT(nmb,8), 20) / INT(nbytes,8))
        IF (0.GE.k) k = 1

        WRITE(*, "(3(A,I0),2(A,I0),A,I0,A)")                            &
     &    "m=", m, " n=", n, " k=", k, " ldi=", ldi, " ldo=", ldo,      &
     &    " size=", INT(k,8) * INT(nbytes,8) / ISHFT(1, 20), "MB"
        CALL libxsmm_init()

        ALLOCATE(a1(ldi*n*k), b1(ldo*n*k))
        an(1:ldi,1:n, 1:k) => a1
        bn(1:ldo,1:n, 1:k) => b1

        !$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(h, i, j) SHARED(n, k, ldi, ldo, an, bn, check)
        DO h = 1, k
          DO j = 1, n
            DO i = 1, ldi
              an(i,j,h) = initial_value(i-1, j-1, ldi*h)
            END DO
            DO i = 1, MAX(MIN(check,1),ldo)
              bn(i,j,h) = X
            END DO
          END DO
        END DO
        !$OMP END PARALLEL DO

        duration = 0D0
        ! matcopy bandwidth assumes RFO in case of copy
        WRITE(*, "(A)") REPEAT("-", W)
        DO r = 1, nrepeat
          IF (0.NE.zero) THEN
            start = libxsmm_timer_tick()
            DO h = 1, k
              CALL libxsmm_matcopy(bn(:,:,h), m=m,n=n, ldi=ldi,ldo=ldo)
            END DO
            d = libxsmm_timer_duration(start, libxsmm_timer_tick())
            IF (0.GE.d) THEN
              duration = 0D0
              EXIT
            END IF
            duration(1) = duration(1) + d
            IF (0.NE.check) THEN
              WRITE(*, "(A,F10.1,A,1A,F10.1,A)") "LIBXSMM (zero):", 1D3 &
     &          * d, " ms", CHAR(9), REAL(1 * k, 8) * REAL(nbytes, 8)   &
     &          / (REAL(ISHFT(1, 20), 8) * d), " MB/s"
            END IF
          END IF

          IF ((0.EQ.zero).OR.(1.LT.zero)) THEN
            start = libxsmm_timer_tick()
            DO h = 1, k
              !CALL libxsmm_matcopy(ptr(bn(:,:,h)), ptr(an(:,:,h)), S,   &
              CALL libxsmm_matcopy(bn(:,:,h), an(:,:,h),                &
     &        m, n, ldi, ldo)
            END DO
            d = libxsmm_timer_duration(start, libxsmm_timer_tick())
            IF ((0.GE.d).OR.(0.LT.diff(check, an, bn, m))) THEN
              duration = 0D0
              EXIT
            END IF
            duration(2) = duration(2) + d
            IF (0.NE.check) THEN
              WRITE(*, "(A,F10.1,A,1A,F10.1,A)") "LIBXSMM (copy):", 1D3 &
     &          * d, " ms", CHAR(9), REAL(3 * k, 8) * REAL(nbytes, 8)   &
     &          / (REAL(ISHFT(1, 20), 8) * d), " MB/s"
            END IF
          END IF
          ! skip non-LIBXSMM measurements
          IF (0.EQ.check) CYCLE

          IF (0.NE.zero) THEN
            start = libxsmm_timer_tick()
            DO h = 1, k
              bn(1:m,:,h) = REAL(0,T)
            END DO
            d = libxsmm_timer_duration(start, libxsmm_timer_tick())
            IF (0.GE.d) THEN
              duration = 0D0
              EXIT
            END IF
            duration(3) = duration(3) + d
            WRITE(*, "(A,F10.1,A,1A,F10.1,A)") "FORTRAN (zero):", 1D3   &
     &        * d, " ms", CHAR(9), REAL(1 * k, 8) * REAL(nbytes, 8)     &
     &        / (REAL(ISHFT(1, 20), 8) * d), " MB/s"
          END IF

          IF ((0.EQ.zero).OR.(1.LT.zero)) THEN
            start = libxsmm_timer_tick()
            DO h = 1, k
              bn(1:m,:,h) = an(1:m,:,h)
            END DO
            d = libxsmm_timer_duration(start, libxsmm_timer_tick())
            IF ((0.GE.d).OR.(0.LT.diff(check, an, bn, m))) THEN
              duration = 0D0
              EXIT
            END IF
            duration(4) = duration(4) + d
            WRITE(*, "(A,F10.1,A,1A,F10.1,A)") "FORTRAN (copy):", 1D3   &
     &        * d, " ms", CHAR(9), REAL(3 * k, 8) * REAL(nbytes, 8)     &
     &        / (REAL(ISHFT(1, 20), 8) * d), " MB/s"
          END IF

          WRITE(*, "(A)") REPEAT("-", W)
        END DO

        DEALLOCATE(a1, b1)

        IF (ANY(0.LT.duration)) THEN
          IF ((1.LT.nrepeat).OR.(0.EQ.check)) THEN
            IF (1.LT.nrepeat) THEN
              WRITE(*, "(A,I0,A)") "Arithmetic average of ",            &
     &          nrepeat, " iterations"
              WRITE(*, "(A)") REPEAT("-", W)
            END IF
            IF (0.LT.duration(1)) THEN
              WRITE(*, "(A,F10.1,A)") "LIBXSMM (zero):",                &
     &          (REAL(1*k*nrepeat, 8) * REAL(nbytes, 8))                &
     &        / (REAL(ISHFT(1, 20), 8) * duration(1)), " MB/s"
            END IF
            IF (0.LT.duration(2)) THEN
              WRITE(*, "(A,F10.1,A)") "LIBXSMM (copy):",                &
     &          (REAL(3*k*nrepeat, 8) * REAL(nbytes, 8))                &
     &        / (REAL(ISHFT(1, 20), 8) * duration(2)), " MB/s"
            END IF
            IF (0.LT.duration(3)) THEN
              WRITE(*, "(A,F10.1,A)") "FORTRAN (zero):",                &
     &          (REAL(1*k*nrepeat, 8) * REAL(nbytes, 8))                &
     &        / (REAL(ISHFT(1, 20), 8) * duration(3)), " MB/s"
            END IF
            IF (0.LT.duration(4)) THEN
              WRITE(*, "(A,F10.1,A)") "FORTRAN (copy):",                &
     &          (REAL(3*k*nrepeat, 8) * REAL(nbytes, 8))                &
     &        / (REAL(ISHFT(1, 20), 8) * duration(4)), " MB/s"
            END IF
            WRITE(*, "(A)") REPEAT("-", W)
          END IF
        ELSE
          WRITE(*, "(A)") "Failed!"
        END IF

      CONTAINS
        PURE REAL(T) FUNCTION initial_value(i, j, m)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: i, j, m
          initial_value = REAL(j * m + i, T)
        END FUNCTION

        PURE REAL(T) FUNCTION diff(check, a, b, m)
          REAL(T), INTENT(IN) :: a(:,:,:), b(:,:,:)
          INTEGER, INTENT(IN) :: check
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND) :: h, i, j
          diff = REAL(0,T)
          IF (0.NE.check) THEN
            DO h = LBOUND(a,3), UBOUND(a,3)
              DO j = LBOUND(a,2), UBOUND(a,2)
                DO i = LBOUND(a,1), m
                  diff = MAX(diff, ABS(b(i,j,h) - a(i,j,h)))
                END DO
                DO i = m+1, UBOUND(b,1)
                  diff = MAX(diff, ABS(b(i,j,h) - X))
                END DO
              END DO
            END DO
          END IF
        END FUNCTION
      END PROGRAM

