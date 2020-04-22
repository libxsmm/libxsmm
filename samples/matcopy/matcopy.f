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
     &                        ptr => libxsmm_ptr
        IMPLICIT NONE

        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER, PARAMETER :: S = 8
        REAL(T), PARAMETER :: X = REAL(-1, T)

        REAL(T), ALLOCATABLE, TARGET :: a1(:), b1(:)
        !DIR$ ATTRIBUTES ALIGN:64 :: a1, b1
        INTEGER(LIBXSMM_BLASINT_KIND) :: m, n, ldi, ldo, i, j, k
        REAL(T), POINTER :: an(:,:), bn(:,:)
        DOUBLE PRECISION :: duration
        INTEGER(8) :: nbytes, start
        INTEGER :: nrepeat
        REAL(T) :: diff

        CHARACTER(32) :: argv
        INTEGER :: argc

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
        ELSE
          ldi = m
        END IF
        IF (4 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(4, argv)
          READ(argv, "(I32)") ldo
        ELSE
          ldo = ldi
        END IF
        IF (5 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(5, argv)
          READ(argv, "(I32)") nrepeat
        ELSE
          nrepeat = 3
        END IF

        nbytes = INT(m * n, 8) * T ! size in Byte
        WRITE(*, "(2(A,I0),2(A,I0),A,I0,A)")                            &
     &    "m=", m, " n=", n, " ldi=", ldi, " ldo=", ldo,                &
     &    " size=", (nbytes / ISHFT(1, 20)), "MB"

        ALLOCATE(a1(ldi*n), b1(ldo*n))
        an(1:ldi,1:n) => a1
        bn(1:ldo,1:n) => b1

        !$OMP PARALLEL DO PRIVATE(i, j) DEFAULT(NONE) SHARED(m, n, an)
        DO j = 1, n
          DO i = 1, ldi
            an(i,j) = initial_value(i - 1, j - 1, ldi)
          END DO
          DO i = 1, ldo
            bn(i,j) = X
          END DO
        END DO
        !$OMP END PARALLEL DO

        start = libxsmm_timer_tick()
        DO k = 1, nrepeat
          CALL libxsmm_matcopy_omp(ptr(b1), ptr(a1), S, m, n, ldi, ldo)
          CALL libxsmm_matcopy(ptr(b1), ptr(a1), S, m, n, ldi, ldo)
          CALL libxsmm_matcopy(bn, an, m, n, ldi, ldo)
          CALL libxsmm_matcopy(b1, a1, m, n, ldi, ldo)
        END DO
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick())

        diff = REAL(0, T)
        DO j = 1, n
          DO i = 1, m
            diff = MAX(diff, ABS(bn(i,j) - an(i,j)))
          END DO
          DO i = m+1, ldo
            diff = MAX(diff, ABS(bn(i,j) - X))
          END DO
        END DO
        DEALLOCATE(a1, b1)

        IF (0.GE.diff) THEN
          IF ((0.LT.duration).AND.(0.LT.nrepeat)) THEN
            ! matcopy bandwidth assumes RFO
            WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ", 3D0     &
     &        * REAL(nbytes, T)                                         &
     &        * REAL(nrepeat, T) / (duration * REAL(ISHFT(1_8, 30), T)),&
     &        " GB/s"
            WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ",         &
     &        1D3 * duration / REAL(nrepeat, T),                        &
     &        " ms"
          END IF
        ELSE
          WRITE(*,*) "Validation failed!"
          STOP 1
        END IF

      CONTAINS
        PURE REAL(T) FUNCTION initial_value(i, j, m)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: i, j, m
          initial_value = REAL(j * m + i, T)
        END FUNCTION
      END PROGRAM

