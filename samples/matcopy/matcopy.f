!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXSMM library.                             !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/libxsmm/libxsmm/              !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!
! Hans Pabst (Intel Corp.)
!=======================================================================!

      PROGRAM matcopy
        USE :: LIBXSMM, ONLY: LIBXSMM_BLASINT_KIND,                     &
     &                        libxsmm_timer_duration,                   &
     &                        libxsmm_timer_tick,                       &
     &                        libxsmm_init,                             &
     &                        xcopy => libxsmm_xmatcopy,                &
     &                        ptr0 => libxsmm_ptr_null,                 &
     &                        ptr => libxsmm_ptr
        IMPLICIT NONE

        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER, PARAMETER :: S = T
        INTEGER, PARAMETER :: W = 50
        REAL(T), PARAMETER :: X = REAL(-1, T) ! pattern
        REAL(T), PARAMETER :: Z = REAL( 0, T) ! zero

        REAL(T), ALLOCATABLE, TARGET :: a1(:), b1(:)
        !DIR$ ATTRIBUTES ALIGN:64 :: a1, b1
        INTEGER(LIBXSMM_BLASINT_KIND) :: m, n, ldi, ldo, h, i, j
        REAL(T), POINTER :: an(:,:,:), bn(:,:,:)
        DOUBLE PRECISION :: d, duration(4)
        INTEGER(8) :: start
        INTEGER :: r, nrepeat, ncount, error
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
          nrepeat = 6
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

        !$OMP PARALLEL DO DEFAULT(NONE) PRIVATE(h, i, j)                &
        !$OMP             SHARED(n, k, ldi, ldo, an, bn, check)
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

        error = 0
        duration = 0D0
        ! matcopy bandwidth assumes NTS in case of copy
        WRITE(*, "(A)") REPEAT("-", W)
        DO r = 1, nrepeat
          IF (0.NE.zero) THEN
            start = libxsmm_timer_tick()
            DO h = 1, k
              !CALL libxsmm_xmatcopy(bn(:,:,h), m=m,n=n, ldi=ldi,ldo=ldo)
              CALL xcopy(ptr(bn(:,:,h)), ptr0(), S, m, n, ldi, ldo)
            END DO
            d = libxsmm_timer_duration(start, libxsmm_timer_tick())
            IF ((0.GE.d).OR.(0.LT.diff(check, m, bn))) THEN
              error = 1
              EXIT
            END IF
            IF (1.LT.r) duration(1) = duration(1) + d
            IF (0.NE.check) THEN
              WRITE(*, "(A,F10.1,A,1A,F10.1,A)") "LIBXSMM (zero):", 1D3 &
     &          * d, " ms", CHAR(9), REAL(1 * k, 8) * REAL(nbytes, 8)   &
     &          / (REAL(ISHFT(1, 20), 8) * d), " MB/s"
            END IF
          END IF

          IF ((0.EQ.zero).OR.(1.LT.zero)) THEN
            start = libxsmm_timer_tick()
            DO h = 1, k
              !CALL libxsmm_xmatcopy(bn(:,:,h), an(:,:,h), m,n, ldi,ldo)
              CALL xcopy(ptr(bn(:,:,h)), ptr(an(:,:,h)),                &
     &          S, m, n, ldi, ldo)
            END DO
            d = libxsmm_timer_duration(start, libxsmm_timer_tick())
            IF ((0.GE.d).OR.(0.LT.diff(check, m, bn, an))) THEN
              error = 2
              EXIT
            END IF
            IF (1.LT.r) duration(2) = duration(2) + d
            IF (0.NE.check) THEN
              WRITE(*, "(A,F10.1,A,1A,F10.1,A)") "LIBXSMM (copy):", 1D3 &
     &          * d, " ms", CHAR(9), REAL(2 * k, 8) * REAL(nbytes, 8)   &
     &          / (REAL(ISHFT(1, 20), 8) * d), " MB/s"
            END IF
          END IF
          ! skip non-LIBXSMM measurements
          IF (0.EQ.check) CYCLE

          IF (0.NE.zero) THEN
            start = libxsmm_timer_tick()
            DO h = 1, k
              bn(1:m,:,h) = Z
            END DO
            d = libxsmm_timer_duration(start, libxsmm_timer_tick())
            IF ((0.GE.d).OR.(0.LT.diff(check, m, bn))) THEN
              error = 3
              EXIT
            END IF
            IF (1.LT.r) duration(3) = duration(3) + d
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
            IF ((0.GE.d).OR.(0.LT.diff(check, m, bn, an))) THEN
              error = 4
              EXIT
            END IF
            IF (1.LT.r) duration(4) = duration(4) + d
            WRITE(*, "(A,F10.1,A,1A,F10.1,A)") "FORTRAN (copy):", 1D3   &
     &        * d, " ms", CHAR(9), REAL(2 * k, 8) * REAL(nbytes, 8)     &
     &        / (REAL(ISHFT(1, 20), 8) * d), " MB/s"
          END IF

          WRITE(*, "(A)") REPEAT("-", W)
        END DO

        DEALLOCATE(a1, b1)

        IF (0.EQ.error) THEN
          IF ((1.LT.nrepeat).OR.(0.EQ.check)) THEN
            ncount = MERGE(nrepeat - 1, nrepeat, 2.LT.nrepeat)
            IF (1.LT.ncount) THEN
              WRITE(*, "(A,I0,A)") "Arithmetic average of ",            &
     &          ncount, " iterations"
              WRITE(*, "(A)") REPEAT("-", W)
            END IF
            IF (0.LT.duration(1)) THEN
              WRITE(*, "(A,F10.1,A)") "LIBXSMM (zero):",                &
     &          (REAL(1*k*ncount, 8) * REAL(nbytes, 8))                 &
     &        / (REAL(ISHFT(1, 20), 8) * duration(1)), " MB/s"
            END IF
            IF (0.LT.duration(2)) THEN
              WRITE(*, "(A,F10.1,A)") "LIBXSMM (copy):",                &
     &          (REAL(2*k*ncount, 8) * REAL(nbytes, 8))                 &
     &        / (REAL(ISHFT(1, 20), 8) * duration(2)), " MB/s"
            END IF
            IF (0.LT.duration(3)) THEN
              WRITE(*, "(A,F10.1,A)") "FORTRAN (zero):",                &
     &          (REAL(1*k*ncount, 8) * REAL(nbytes, 8))                 &
     &        / (REAL(ISHFT(1, 20), 8) * duration(3)), " MB/s"
            END IF
            IF (0.LT.duration(4)) THEN
              WRITE(*, "(A,F10.1,A)") "FORTRAN (copy):",                &
     &          (REAL(2*k*ncount, 8) * REAL(nbytes, 8))                 &
     &        / (REAL(ISHFT(1, 20), 8) * duration(4)), " MB/s"
            END IF
            WRITE(*, "(A)") REPEAT("-", W)
          END IF
        ELSE
          SELECT CASE (error)
            CASE (1)
              WRITE(*, "(A)") "Error: LIBXSMM-zero failed!"
            CASE (2)
              WRITE(*, "(A)") "Error: LIBXSMM-copy failed!"
            CASE (3)
              WRITE(*, "(A)") "Error: FORTRAN-zero failed!"
            CASE (4)
              WRITE(*, "(A)") "Error: FORTRAN-copy failed!"
            CASE DEFAULT
              WRITE(*, "(A)") "Unknown error!"
          END SELECT
        END IF

      CONTAINS
        PURE REAL(T) FUNCTION initial_value(i, j, m)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: i, j, m
          initial_value = REAL(j * m + i, T)
        END FUNCTION

        PURE REAL(T) FUNCTION diff(check, m, mat, ref)
          INTEGER, INTENT(IN) :: check
          REAL(T), INTENT(IN) :: mat(:,:,:)
          REAL(T), INTENT(IN), OPTIONAL :: ref(:,:,:)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND) :: h, i, j
          diff = Z
          IF (0.NE.check) THEN
            DO h = LBOUND(mat,3), UBOUND(mat,3)
              DO j = LBOUND(mat,2), UBOUND(mat,2)
                DO i = LBOUND(mat,1), m
                  IF (PRESENT(ref)) THEN ! copy
                    diff = MAX(diff, ABS(mat(i,j,h) - ref(i,j,h)))
                  ELSE ! zero
                    diff = MAX(diff, ABS(mat(i,j,h) - Z))
                  END IF
                END DO
                DO i = m+1, UBOUND(mat,1)
                  diff = MAX(diff, ABS(mat(i,j,h) - X))
                END DO
              END DO
            END DO
          END IF
        END FUNCTION
      END PROGRAM

