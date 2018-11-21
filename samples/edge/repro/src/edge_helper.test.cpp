#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include "edge_helper_depricated.hpp"

int main( int i_argc, char *i_argv[] ) {
  std::cout << "Testing *readSparseMatrixCsc*" << std::endl;
  {
    std::string l_cscFileName[] =
    {
      "../mats/tet4_4_stiffT_2_csc.mtx",
      "../mats/tet4_4_fluxL_2_csc.mtx",
      "../mats/tet4_4_fluxT_2_csc.mtx"
    };

    unsigned int l_cscMatSize[][3] =
    {
      {35, 35, 287},
      {35, 15, 363},
      {15, 35, 363}
    };

    std::vector< double >       l_matVal;
    std::vector< unsigned int > l_matColPtr;
    std::vector< unsigned int > l_matRowIdx;

    for ( unsigned int l_i = 0; l_i < 3; l_i++ ) {
      edge::reproducers::
      readSparseMatrixCsc( l_cscFileName[l_i],
                           l_matVal,
                           l_matColPtr,
                           l_matRowIdx );
      assert( l_matVal.size() == l_cscMatSize[l_i][2] );
      assert( l_matColPtr.size() == l_cscMatSize[l_i][1]+1 );
    }
  }
  std::cout << "(3) tests passed" << std::endl;

  std::cout << "Testing *readSparseMatrixCsr*" << std::endl;
  {
    std::string l_csrFileName[] =
    {
      "../mats/tet4_4_stiffT_2_csr.mtx",
      "../mats/tet4_4_fluxL_2_csr.mtx",
      "../mats/tet4_4_fluxT_2_csr.mtx"
    };

    unsigned int l_csrMatSize[][3] =
    {
      {35, 35, 287},
      {35, 15, 363},
      {15, 35, 363}
    };

    std::vector< double >       l_matVal;
    std::vector< unsigned int > l_matRowPtr;
    std::vector< unsigned int > l_matColIdx;

    for ( unsigned int l_i = 0; l_i < 3; l_i++ ) {
      edge::reproducers::
      readSparseMatrixCsr( l_csrFileName[l_i],
                           l_matVal,
                           l_matRowPtr,
                           l_matColIdx );
      assert( l_matVal.size() == l_csrMatSize[l_i][2] );
      assert( l_matRowPtr.size() == l_csrMatSize[l_i][0]+1 );
    }
  }
  std::cout << "(3) tests passed" << std::endl;

  std::cout << "Testing *selectSubSparseMatrixCsc*" << std::endl;
  {
    std::string l_cscFileName[] =
    {
      "../mats/tet4_4_stiffT_2_csc.mtx",
      "../mats/tet4_4_stiffT_2_csc.mtx",
      "../mats/tet4_4_stiffV_2_csc.mtx",
      "../mats/tet4_4_stiffV_2_csc.mtx",
      "../mats/tet4_4_stiffV_2_csc.mtx"
    };

    unsigned int l_subMat[][2] =
    {
      {35, 20},
      {20, 35},
      {35, 20},
      {20, 35},
      {35, 35}
    };

    unsigned int l_cscSubMatSize[][3] =
    {
      {35, 20, 287},
      {20, 35, 92},
      {35, 20, 92},
      {20, 35, 287},
      {35, 35, 287}
    };

    std::vector< double >    l_matVal[2];
    std::vector< unsigned int > l_matColPtr[2];
    std::vector< unsigned int > l_matRowIdx[2];

    for ( unsigned int l_i = 0; l_i < 5; l_i++ ) {
      edge::reproducers::
      readSparseMatrixCsc( l_cscFileName[l_i],
                           l_matVal[0],
                           l_matColPtr[0],
                           l_matRowIdx[0] );
      edge::reproducers::
      selectSubSparseMatrixCsc( l_matVal[0], l_matColPtr[0], l_matRowIdx[0],
                                l_subMat[l_i][0], l_subMat[l_i][1],
                                l_matVal[1], l_matColPtr[1], l_matRowIdx[1] );

      assert( l_matVal[1].size() == l_cscSubMatSize[l_i][2] );
      assert( l_matColPtr[1].size() == l_cscSubMatSize[l_i][1]+1 );
    }
  }
  std::cout << "(5) tests passed" << std::endl;

  return 0;
}
