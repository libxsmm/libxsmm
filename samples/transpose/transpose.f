!*****************************************************************************!
!* Copyright (c) 2016-2017, Intel Corporation                                *!
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
!* Hans Pabst (Intel Corp.)                                                  *!
!*****************************************************************************!

      PROGRAM transpose
        USE :: LIBXSMM
        IMPLICIT NONE

        INTEGER, PARAMETER :: T = KIND(0D0)

        REAL(T), ALLOCATABLE, TARGET :: a(:,:), b(:,:)
        !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNMENT :: a, b
        INTEGER(LIBXSMM_BLASINT_KIND) :: m, n, lda, ldb, i, j, size
        DOUBLE PRECISION :: duration
        INTEGER(8) :: start
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

        size = m * n * T ! size in Byte
        WRITE(*, "(2(A,I0),2(A,I0),A,I0,A,2A,2A,1A)")                   &
     &    "m=", m, " n=", n, " lda=", lda, " ldb=", ldb,                &
     &    " size=", (size / ISHFT(1, 20)),                              &
     &    "MB (", MERGE("DP", "SP", 8.EQ.T), ", ",                      &
     &    TRIM(MERGE("out-of-place", "in-place    ", 'o'.EQ.trans)), ")"

        ! Allocate matrices
        ALLOCATE(a(lda,MERGE(n,lda,('o'.EQ.trans).OR.('O'.EQ.trans))))
        ALLOCATE(b(ldb,MERGE(m,ldb,('o'.EQ.trans).OR.('O'.EQ.trans))))

        DO j = 1, n
          DO i = 1, m
            a(i,j) = initial_value(i - 1, j - 1, m)
          END DO
        END DO

        IF (('o'.EQ.trans).OR.('O'.EQ.trans)) THEN
          start = libxsmm_timer_tick();
          CALL libxsmm_otrans(C_LOC(b), C_LOC(a), T, m, n, lda, ldb)
          !CALL libxsmm_otrans(C_LOC(a), C_LOC(b), T, n, m, ldb, lda)
          CALL libxsmm_dotrans(a, b, n, m)
          duration = libxsmm_timer_duration(                            &
     &                  start, libxsmm_timer_tick());
        ELSE ! in-place
          start = libxsmm_timer_tick();
          ! TODO: in-place
          duration = libxsmm_timer_duration(                            &
     &                  start, libxsmm_timer_tick());
        END IF

        diff = 0
        DO j = 1, n
          DO i = 1, m
            diff = MAX(diff,                                            &
     &                ABS(a(i,j) - initial_value(i - 1, j - 1, m)))
          END DO
        END DO

        ! Deallocate matrices
        DEALLOCATE(a)
        DEALLOCATE(b)

        IF (0.EQ.diff) THEN
          IF (0.LT.duration) THEN
            WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ",         &
     &        size / (duration * ISHFT(1_8, 30)), " GB/s"
          END IF
          WRITE(*, "(1A,A,F10.1,A)") CHAR(9),                           &
     &        "duration:   ", 1D3 * duration, " ms"
        ELSE
          WRITE(*,*) "Validation failed!"
          STOP 1
        END IF

      CONTAINS
        PURE REAL(T) FUNCTION initial_value(i, j, ld)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: i, j, ld
          initial_value = j * ld + i
        END FUNCTION
      END PROGRAM

