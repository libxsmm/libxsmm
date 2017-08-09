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

        REAL(T), ALLOCATABLE, TARGET :: a1(:), b1(:)
        !DIR$ ATTRIBUTES ALIGN:LIBXSMM_ALIGNMENT :: a1, b1
        INTEGER(LIBXSMM_BLASINT_KIND) :: m, n, lda, ldb, i, j, k
        REAL(T), POINTER :: an(:,:), bn(:,:), bt(:,:)
        DOUBLE PRECISION :: duration
        INTEGER(8) :: size, start
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

        size = INT(m * n, 8) * T ! size in Byte
        WRITE(*, "(2(A,I0),2(A,I0),A,I0,A,2A,2A,1A)")                   &
     &    "m=", m, " n=", n, " ldi=", lda, " ldo=", ldb,                &
     &    " size=", (size / ISHFT(1, 20)),                              &
     &    "MB (", MERGE("DP", "SP", 8.EQ.T), ", ",                      &
     &    TRIM(MERGE("out-of-place", "in-place    ",                    &
     &      ('o'.EQ.trans).OR.('O'.EQ.trans))), ")"

        ALLOCATE(b1(ldb*MAX(m,n)))
        bn(1:ldb,1:n) => b1
        bt(1:ldb,1:m) => b1

        IF (('o'.EQ.trans).OR.('O'.EQ.trans)) THEN
          ALLOCATE(a1(lda*n))
          an(1:lda,1:n) => a1
          DO j = 1, n
            DO i = 1, m
              an(i,j) = initial_value(i - 1, j - 1, m)
            END DO
          END DO
          start = libxsmm_timer_tick()
          DO k = 1, nrepeat
            CALL libxsmm_otrans_omp(C_LOC(b1), C_LOC(a1),               &
     &              T, m, n, lda, ldb)
          END DO
          duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
          DEALLOCATE(a1)
        ELSE ! in-place
          DO j = 1, n
            DO i = 1, m
              bn(i,j) = initial_value(i - 1, j - 1, m)
            END DO
          END DO
          start = libxsmm_timer_tick()
          DO k = 1, nrepeat
            CALL libxsmm_itrans(C_LOC(b1), T, m, n, ldb)
          END DO
          duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
        END IF

        diff = 0
        DO j = 1, n
          DO i = 1, m
            diff = MAX(diff,                                            &
     &                ABS(bt(j,i) - initial_value(i - 1, j - 1, m)))
          END DO
        END DO
        DEALLOCATE(b1)

        IF (0.EQ.diff) THEN
          IF ((0.LT.duration).AND.(0.LT.nrepeat)) THEN
            ! out-of-place transpose bandwidth assumes RFO
            WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "bandwidth:  ",         &
     &        MERGE(3_8, 2_8, ('o'.EQ.trans).OR.('O'.EQ.trans))         &
     &          * size * nrepeat / (duration * ISHFT(1_8, 30)), " GB/s"
            WRITE(*, "(1A,A,F10.1,A)") CHAR(9), "duration:   ",         &
     &        1D3 * duration / nrepeat, " ms"
          END IF
        ELSE
          WRITE(*,*) "Validation failed!"
          STOP 1
        END IF

      CONTAINS
        PURE REAL(T) FUNCTION initial_value(i, j, m)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: i, j, m
          initial_value = j * m + i
        END FUNCTION
      END PROGRAM

