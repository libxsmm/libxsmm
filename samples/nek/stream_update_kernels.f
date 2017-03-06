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
!* Alexander Heinecke (Intel Corp.)                                          *!
!*****************************************************************************!

      MODULE STREAM_UPDATE_KERNELS
        USE, INTRINSIC :: ISO_C_BINDING
        IMPLICIT NONE

        INTERFACE
          SUBROUTINE stream_update_helmholtz( i_g1, i_g2, i_g3,         &
     &                                        i_tm1, i_tm2, i_tm3,      &
     &                                        i_a, i_b, io_c,           &
     &                                        i_h1, i_h2, i_length )    &
     &    BIND(C, name='stream_update_helmholtz')
            IMPORT :: C_DOUBLE, C_INT
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g1
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g2
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g3
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm1
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm2
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm3
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_a
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_b
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(INOUT) :: io_c
            REAL(KIND=C_DOUBLE), VALUE,        INTENT(IN)    :: i_h1
            REAL(KIND=C_DOUBLE), VALUE,        INTENT(IN)    :: i_h2
            INTEGER(C_INT),      VALUE,        INTENT(IN)    :: i_length
          END SUBROUTINE

          SUBROUTINE stream_update_helmholtz_no_h2(                     &
     &                                        i_g1, i_g2, i_g3,         &
     &                                        i_tm1, i_tm2, i_tm3,      &
     &                                        io_c, i_h1, i_length )    &
     &    BIND(C, name='stream_update_helmholtz_no_h2')
            IMPORT :: C_DOUBLE, C_INT
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g1
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g2
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g3
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm1
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm2
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm3
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(INOUT) :: io_c
            REAL(KIND=C_DOUBLE), VALUE,        INTENT(IN)    :: i_h1
            INTEGER(C_INT),      VALUE,        INTENT(IN)    :: i_length
          END SUBROUTINE

          SUBROUTINE stream_update_var_helmholtz(                       &
     &                                        i_g1, i_g2, i_g3,         &
     &                                        i_tm1, i_tm2, i_tm3,      &
     &                                        i_a, i_b, io_c,           &
     &                                        i_h1, i_h2, i_length )    &
     &    BIND(C, name='stream_update_var_helmholtz')
            IMPORT :: C_DOUBLE, C_INT
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g1
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g2
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g3
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm1
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm2
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm3
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_a
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_b
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(INOUT) :: io_c
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_h1
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_h2
            INTEGER(C_INT),      VALUE,        INTENT(IN)    :: i_length
          END SUBROUTINE

          SUBROUTINE stream_vector_compscale( i_a, i_b, io_c,           &
     &                                        i_length )                &
     &    BIND(C, name='stream_vector_compscale')
            IMPORT :: C_DOUBLE, C_INT
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_a
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_b
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(INOUT) :: io_c
            INTEGER(C_INT),      VALUE,        INTENT(IN)    :: i_length
          END SUBROUTINE

          SUBROUTINE stream_vector_copy( i_a, io_c, i_length )          &
     &    BIND(C, name='stream_vector_copy')
            IMPORT :: C_DOUBLE, C_INT
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_a
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(INOUT) :: io_c
            INTEGER(C_INT),      VALUE,        INTENT(IN)    :: i_length
          END SUBROUTINE

          SUBROUTINE stream_vector_set( i_scalar, io_c, i_length )      &
     &    BIND(C, name='stream_vector_set')
            IMPORT :: C_DOUBLE, C_INT
            REAL(KIND=C_DOUBLE), VALUE,        INTENT(IN)    :: i_scalar
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(INOUT) :: io_c
            INTEGER(C_INT),      VALUE,        INTENT(IN)    :: i_length
          END SUBROUTINE
        END INTERFACE
      END MODULE

