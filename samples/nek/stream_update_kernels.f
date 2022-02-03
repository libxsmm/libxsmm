!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXSMM library.                             !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/libxsmm/libxsmm/              !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!
! Alexander Heinecke (Intel Corp.)
!=======================================================================!

      MODULE STREAM_UPDATE_KERNELS
        USE, INTRINSIC :: ISO_C_BINDING
        IMPLICIT NONE

        INTERFACE
          SUBROUTINE stream_update_helmholtz( i_g1, i_g2, i_g3,         &
     &                                        i_tm1, i_tm2, i_tm3,      &
     &                                        i_a, i_b, io_c,           &
     &                                        i_h1, i_h2, i_length )
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
            REAL(KIND=C_DOUBLE),               INTENT(IN)    :: i_h1
            REAL(KIND=C_DOUBLE),               INTENT(IN)    :: i_h2
            INTEGER(C_INT),                    INTENT(IN)    :: i_length
          END SUBROUTINE

          SUBROUTINE stream_update_helmholtz_no_h2(                     &
     &                                        i_g1, i_g2, i_g3,         &
     &                                        i_tm1, i_tm2, i_tm3,      &
     &                                        io_c, i_h1, i_length )
            IMPORT :: C_DOUBLE, C_INT
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g1
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g2
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_g3
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm1
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm2
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_tm3
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(INOUT) :: io_c
            REAL(KIND=C_DOUBLE),               INTENT(IN)    :: i_h1
            INTEGER(C_INT),                    INTENT(IN)    :: i_length
          END SUBROUTINE

          SUBROUTINE stream_update_var_helmholtz(                       &
     &                                        i_g1, i_g2, i_g3,         &
     &                                        i_tm1, i_tm2, i_tm3,      &
     &                                        i_a, i_b, io_c,           &
     &                                        i_h1, i_h2, i_length )
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
            INTEGER(C_INT),                    INTENT(IN)    :: i_length
          END SUBROUTINE

          SUBROUTINE stream_vector_compscale( i_a, i_b, io_c, i_length )
            IMPORT :: C_DOUBLE, C_INT
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_a
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_b
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(INOUT) :: io_c
            INTEGER(C_INT),                    INTENT(IN)    :: i_length
          END SUBROUTINE

          SUBROUTINE stream_vector_copy( i_a, io_c, i_length )
            IMPORT :: C_DOUBLE, C_INT
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(IN)    :: i_a
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(INOUT) :: io_c
            INTEGER(C_INT),                    INTENT(IN)    :: i_length
          END SUBROUTINE

          SUBROUTINE stream_vector_set( i_scalar, io_c, i_length )
            IMPORT :: C_DOUBLE, C_INT
            REAL(KIND=C_DOUBLE),               INTENT(IN)    :: i_scalar
            REAL(KIND=C_DOUBLE), DIMENSION(*), INTENT(INOUT) :: io_c
            INTEGER(C_INT),                    INTENT(IN)    :: i_length
          END SUBROUTINE
        END INTERFACE
      END MODULE

