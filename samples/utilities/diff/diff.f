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

      PROGRAM diff
        USE :: LIBXSMM, ONLY: LIBXSMM_TICKINT_KIND,                     &
     &                        libxsmm_timer_duration,                   &
     &                        libxsmm_timer_tick,                       &
     &                        libxsmm_init,                             &
     &                        libxsmm_diff
        IMPLICIT NONE

        INTEGER, PARAMETER :: W = 34
        INTEGER, PARAMETER :: T = 4

        INTEGER(T), ALLOCATABLE, TARGET :: a(:), b(:)
        !DIR$ ATTRIBUTES ALIGN:64 :: a, b
        INTEGER(LIBXSMM_TICKINT_KIND) :: start
        DOUBLE PRECISION :: duration(3), d
        INTEGER :: i, n, nrepeat
        INTEGER(8) :: nbytes

        CHARACTER(32) :: argv
        INTEGER :: argc

        argc = COMMAND_ARGUMENT_COUNT()
        IF (1 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(1, argv)
          READ(argv, "(I32)") n
        ELSE
          n = 0
        END IF
        IF (2 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(2, argv)
          READ(argv, "(I32)") nrepeat
        ELSE
          nrepeat = 5
        END IF

        duration = 0D0
        n = MERGE(n, ISHFT(ISHFT(2, 20) / T, 10), 0 < n) ! 2 GB by default
        nbytes = INT(n, 8) * T
        WRITE(*, "(A,I0,A,I0,A,A,I0,A)")                                &
     &    "nelements=", n, " typesize=", T, "Byte",                     &
     &    " size=", nbytes / ISHFT(1, 20), "MB"
        CALL libxsmm_init()

        ALLOCATE(a(n), b(n))
        DO i = 1, n
          a(i) = i - 1
          b(i) = i - 1
        END DO

        WRITE(*, "(A)") REPEAT("-", W)
        DO i = 1, nrepeat
          start = libxsmm_timer_tick()
          IF (.NOT. libxsmm_diff(a, b)) THEN
            d = libxsmm_timer_duration(start, libxsmm_timer_tick())
            duration(1) = duration(1) + d
            WRITE(*, "(A,F10.1,A)") "DIFF (LIBXSMM):", 1D3 * d, " ms"
          ELSE
            WRITE(*, "(A)") "Validation failed!"
          END IF

          start = libxsmm_timer_tick()
          IF (ALL(a .EQ. b)) THEN
            d = libxsmm_timer_duration(start, libxsmm_timer_tick())
            duration(2) = duration(2) + d
            WRITE(*, "(A,F10.1,A)") "ALL  (Fortran):", 1D3 * d, " ms"
          ELSE
            WRITE(*, "(A)") "Validation failed!"
          END IF

          start = libxsmm_timer_tick()
          IF (.NOT. ANY(a .NE. b)) THEN
            d = libxsmm_timer_duration(start, libxsmm_timer_tick())
            duration(3) = duration(3) + d
            WRITE(*, "(A,F10.1,A)") "ANY  (Fortran):", 1D3 * d, " ms"
          ELSE
            WRITE(*, "(A)") "Validation failed!"
          END IF
          WRITE(*, "(A)") REPEAT("-", W)
        END DO

        IF (ALL(0 .LT. duration)) THEN
          WRITE(*, "(A,I0,A)") "Arithmetic average of ",                &
     &      nrepeat, " iterations"
          WRITE(*, "(A)") REPEAT("-", W)
          WRITE(*, "(A,F10.1,A)") "DIFF (LIBXSMM):",                    &
     &      REAL(nbytes, 8) * REAL(nrepeat, 8) /                        &
     &      (duration(1) * REAL(ISHFT(1, 20), 8)), " MB/s"
          WRITE(*, "(A,F10.1,A)") "ALL  (Fortran):",                    &
     &      REAL(nbytes, 8) * REAL(nrepeat, 8) /                        &
     &      (duration(2) * REAL(ISHFT(1, 20), 8)), " MB/s"
          WRITE(*, "(A,F10.1,A)") "ANY  (Fortran):",                    &
     &      REAL(nbytes, 8) * REAL(nrepeat, 8) /                        &
     &      (duration(3) * REAL(ISHFT(1, 20), 8)), " MB/s"
          WRITE(*, "(A)") REPEAT("-", W)
        END IF

        DEALLOCATE(a, b)
      END PROGRAM

