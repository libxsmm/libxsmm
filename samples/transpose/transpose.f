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

      PROGRAM transpose
        USE :: LIBXSMM, ONLY: LIBXSMM_BLASINT_KIND,                     &
     &                        libxsmm_timer_duration,                   &
     &                        libxsmm_timer_tick,                       &
     &                        libxsmm_otrans_omp,                       &
     &                        libxsmm_otrans,                           &
     &                        libxsmm_itrans,                           &
     &                        libxsmm_ptr
        IMPLICIT NONE

        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER, PARAMETER :: S = 8

        REAL(T), ALLOCATABLE, TARGET :: a1(:), b1(:)
        !DIR$ ATTRIBUTES ALIGN:64 :: a1, b1
        INTEGER(LIBXSMM_BLASINT_KIND) :: m, n, lda, ldb, i, j, k
        REAL(T), POINTER :: an(:,:), bn(:,:), bt(:,:)
        DOUBLE PRECISION :: duration
        INTEGER(8) :: nbytes, start
        INTEGER :: nrepeat
        REAL(T) :: diff

        CHARACTER(32) :: argv
        CHARACTER :: trans
        INTEGER :: argc

        argc = COMMAND_ARGUMENT_COUNT()
        IF (1 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(1, trans)
        ELSE
          trans = 'o'
        END IF
        IF (2 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(2, argv)
          READ(argv, "(I32)") m
        ELSE
          m = 4096
        END IF
        IF (3 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(3, argv)
          READ(argv, "(I32)") n
        ELSE
          n = m
        END IF
        IF (4 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(4, argv)
          READ(argv, "(I32)") lda
        ELSE
          lda = m
        END IF
        IF (5 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(5, argv)
          READ(argv, "(I32)") ldb
        ELSE
          ldb = n
        END IF
        IF (6 <= argc) THEN
          CALL GET_COMMAND_ARGUMENT(6, argv)
          READ(argv, "(I32)") nrepeat
        ELSE
          nrepeat = 3
        END IF

        nbytes = INT(m * n, 8) * T ! size in Byte
        WRITE(*, "(2(A,I0),2(A,I0),A,I0,A)")                            &
     &    "m=", m, " n=", n, " ldi=", lda, " ldo=", ldb,                &
     &    " size=", (nbytes / ISHFT(1, 20)), "MB"

        ALLOCATE(b1(ldb*MAX(m,n)))
        bn(1:ldb,1:n) => b1
        bt(1:ldb,1:m) => b1

        IF (('o'.EQ.trans).OR.('O'.EQ.trans)) THEN
          ALLOCATE(a1(lda*n))
          an(1:lda,1:n) => a1
          !$OMP PARALLEL DO PRIVATE(i, j) DEFAULT(NONE) SHARED(m, n, an)
          DO j = 1, n
            DO i = 1, m
              an(i,j) = initial_value(i - 1, j - 1, m)
            END DO
          END DO
          !$OMP END PARALLEL DO
          start = libxsmm_timer_tick()
          DO k = 1, nrepeat
            CALL libxsmm_otrans_omp(libxsmm_ptr(b1), libxsmm_ptr(a1),   &
     &              S, m, n, lda, ldb)
          END DO
          duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
          DEALLOCATE(a1)
        ELSE ! in-place
          !$OMP PARALLEL DO PRIVATE(i, j) DEFAULT(NONE) SHARED(m, n, bn)
          DO j = 1, n
            DO i = 1, m
              bn(i,j) = initial_value(i - 1, j - 1, m)
            END DO
          END DO
          !$OMP END PARALLEL DO
          start = libxsmm_timer_tick()
          DO k = 1, nrepeat
            CALL libxsmm_itrans(libxsmm_ptr(b1), S, m, n, ldb)
          END DO
          duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
        END IF

        diff = REAL(0, T)
        DO j = 1, n
          DO i = 1, m
            diff = MAX(diff,                                            &
     &                ABS(bt(j,i) - initial_value(i - 1, j - 1, m)))
          END DO
        END DO
        DEALLOCATE(b1)

        IF (0.GE.diff) THEN
          IF ((0.LT.duration).AND.(0.LT.nrepeat)) THEN
            ! out-of-place transpose bandwidth assumes RFO
            WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ",         &
     &        REAL(nbytes, T)                                           &
     &        * MERGE(3D0, 2D0, ('o'.EQ.trans).OR.('O'.EQ.trans))       &
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

