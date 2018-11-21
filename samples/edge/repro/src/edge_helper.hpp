#ifndef EDGE_REPRODUCER_HELPER_HPP
#define EDGE_REPRODUCER_HELPER_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

namespace edge {
  namespace reproducers {
    template<typename T_REAL>
    int readSparseMatrixCsc( std::string            const   i_fileName,
                             std::vector< T_REAL >        & io_matVal,
                             std::vector< unsigned int >  & io_matColPtr,
                             std::vector< unsigned int >  & io_matRowIdx );

    template<typename T_REAL>
    int readSparseMatrixCsr ( std::string            const   i_fileName,
                              std::vector< T_REAL >        & io_matVal,
                              std::vector< unsigned int >  & io_matRowPtr,
                              std::vector< unsigned int >  & io_matColIdx );

    template<typename T_REAL>
    int readSparseMatrixDense( std::string            const   i_fileName,
                               std::vector< T_REAL >        & io_matVal );

    template<typename T_REAL>
    int selectSubSparseMatrixCsc ( std::vector< T_REAL >       const & i_matVal,
                                   std::vector< unsigned int > const & i_matColPtr,
                                   std::vector< unsigned int > const & i_matRowIdx,
                                   unsigned int                const   i_nSubRows,
                                   unsigned int                const   i_nSubCols,
                                   std::vector< T_REAL >             & o_subMatVal,
                                   std::vector< unsigned int >       & o_subMatColPtr,
                                   std::vector< unsigned int >       & o_subMatRowIdx );
  }
}

template<typename T_REAL>
int edge::reproducers::readSparseMatrixCsc( std::string            const   i_fileName,
                                            std::vector< T_REAL >        & io_matVal,
                                            std::vector< unsigned int >  & io_matColPtr,
                                            std::vector< unsigned int >  & io_matRowIdx ) {
  std::ifstream l_fp( i_fileName );
  std::string l_lineBuf;

  unsigned int l_header = 0;
  unsigned int l_nEntries;
  unsigned int l_nCols;
  unsigned int l_nRows;
  unsigned int l_row;
  unsigned int l_col;
  double       l_entry;
  unsigned int l_nzCounter;
  unsigned int l_colCounter;
  int          l_errCheck;

  while (l_fp) {
    std::getline(l_fp, l_lineBuf);
    if ( l_lineBuf.length() == 0 || l_lineBuf[0] == '%' ) continue;
    if (l_header == 0) {
      l_errCheck= sscanf(l_lineBuf.c_str(), "%u %u %u", &l_nRows, &l_nCols, &l_nEntries);
      if (l_errCheck != 3) abort();

      io_matVal.resize(l_nEntries);
      io_matColPtr.resize(l_nCols+1);
      io_matRowIdx.resize(l_nEntries);

      l_nzCounter = 0;
      l_colCounter = 0;

      l_header = 1;
    } else {
      l_errCheck= sscanf(l_lineBuf.c_str(), "%u %u %lf", &l_row, &l_col, &l_entry);
      if (l_errCheck != 3) abort();

      io_matVal[l_nzCounter] = (T_REAL)l_entry;
      io_matRowIdx[l_nzCounter] = l_row - 1;
      for ( unsigned int l_cc = l_colCounter; l_cc < l_col; l_cc++ ) {
        io_matColPtr[l_cc] = l_nzCounter;
      }
      l_nzCounter++;
      l_colCounter = l_col;
    }
  }
  assert ( l_nzCounter == l_nEntries );

  for ( unsigned int l_cc = l_colCounter; l_cc < l_nCols+1; l_cc++ ) {
        io_matColPtr[l_cc] = l_nzCounter;
  }
  assert ( io_matColPtr.back() == l_nEntries );

  return 0;
};


template<typename T_REAL>
int edge::reproducers::readSparseMatrixCsr( std::string            const   i_fileName,
                                            std::vector< T_REAL >        & io_matVal,
                                            std::vector< unsigned int >  & io_matRowPtr,
                                            std::vector< unsigned int >  & io_matColIdx ) {
  std::ifstream l_fp( i_fileName );
  std::string l_lineBuf;

  unsigned int l_header = 0;
  unsigned int l_nEntries;
  unsigned int l_nCols;
  unsigned int l_nRows;
  unsigned int l_row;
  unsigned int l_col;
  double       l_entry;
  unsigned int l_nzCounter;
  unsigned int l_rowCounter;
  int          l_errCheck;

  while (l_fp) {
    getline(l_fp, l_lineBuf);
    if ( l_lineBuf.length() == 0 || l_lineBuf[0] == '%' ) continue;
    if (l_header == 0) {
      l_errCheck= sscanf(l_lineBuf.c_str(), "%u %u %u", &l_nRows, &l_nCols, &l_nEntries);
      if (l_errCheck != 3) abort();

      io_matVal.resize(l_nEntries);
      io_matRowPtr.resize(l_nRows+1);
      io_matColIdx.resize(l_nEntries);

      l_nzCounter = 0;
      l_rowCounter = 0;

      l_header = 1;
    } else {
      l_errCheck= sscanf(l_lineBuf.c_str(), "%u %u %lf", &l_row, &l_col, &l_entry);
      if (l_errCheck != 3) abort();

      io_matVal[l_nzCounter] = (T_REAL)l_entry;
      io_matColIdx[l_nzCounter] = l_col - 1;
      for ( unsigned int l_rr = l_rowCounter; l_rr < l_row; l_rr++ ) {
        io_matRowPtr[l_rr] = l_nzCounter;
      }
      l_nzCounter++;
      l_rowCounter = l_row;
    }
  }
  assert ( l_nzCounter == l_nEntries );

  for ( unsigned int l_rr = l_rowCounter; l_rr < l_nRows+1; l_rr++ ) {
        io_matRowPtr[l_rr] = l_nzCounter;
  }
  assert ( io_matRowPtr.back() == l_nEntries );

  return 0;
};

template<typename T_REAL>
int edge::reproducers::readSparseMatrixDense( std::string            const   i_fileName,
                                              std::vector< T_REAL >        & io_matVal ) {
  std::ifstream l_fp( i_fileName );
  std::string l_lineBuf;

  unsigned int l_header = 0;
  unsigned int l_nEntries;
  unsigned int l_nCols;
  unsigned int l_nRows;
  unsigned int l_row;
  unsigned int l_col;
  double       l_entry;
  unsigned int l_nzCounter;
  int          l_errCheck;

  while (l_fp) {
    std::getline(l_fp, l_lineBuf);
    if ( l_lineBuf.length() == 0 || l_lineBuf[0] == '%' ) continue;
    if (l_header == 0) {
      l_errCheck= sscanf(l_lineBuf.c_str(), "%u %u %u", &l_nRows, &l_nCols, &l_nEntries);
      if (l_errCheck != 3) abort();

      io_matVal.resize(l_nRows * l_nCols);

      for (unsigned int l_i = 0; l_i < l_nRows; l_i++)
        for (unsigned int l_j = 0; l_j < l_nCols; l_j++)
          io_matVal[l_i*l_nCols+l_j] = 0.0;

      l_header = 1;
    } else {
      l_errCheck= sscanf(l_lineBuf.c_str(), "%u %u %lf", &l_row, &l_col, &l_entry);
      if (l_errCheck != 3) abort();

      io_matVal[(l_row-1)*l_nCols+(l_col-1)] = (T_REAL)l_entry;
      l_nzCounter++;
    }
  }
  assert ( l_nzCounter == l_nEntries );

  return 0;
};

template<typename T_REAL>
int edge::reproducers::selectSubSparseMatrixCsc( std::vector< T_REAL >       const & i_matVal,
                                                 std::vector< unsigned int > const & i_matColPtr,
                                                 std::vector< unsigned int > const & i_matRowIdx,
                                                 unsigned int                const   i_nSubRows,
                                                 unsigned int                const   i_nSubCols,
                                                 std::vector< T_REAL >             & o_subMatVal,
                                                 std::vector< unsigned int >       & o_subMatColPtr,
                                                 std::vector< unsigned int >       & o_subMatRowIdx
                                               ) {
  unsigned int l_nCols = i_matColPtr.size() - 1;
  unsigned int l_nEntries = i_matVal.size();
  unsigned int l_subNzCounter = 0;
  std::vector< T_REAL >       l_tmpMatVal;
  std::vector< unsigned int > l_tmpMatColPtr;
  std::vector< unsigned int > l_tmpMatRowIdx;

  l_tmpMatVal.resize(l_nEntries);
  l_tmpMatColPtr.resize(i_nSubCols+1);
  l_tmpMatRowIdx.resize(l_nEntries);

  for ( unsigned int l_cc = 0; l_cc < l_nCols; l_cc++ ) {
    if ( l_cc >= i_nSubCols ) break;

    l_tmpMatColPtr[l_cc] = l_subNzCounter;
    for ( unsigned int l_nz = i_matColPtr[l_cc]; l_nz < i_matColPtr[l_cc+1]; l_nz++ ) {
      if ( i_matRowIdx[l_nz] >= i_nSubRows ) break;

      l_tmpMatVal[l_subNzCounter] = i_matVal[l_nz];
      l_tmpMatRowIdx[l_subNzCounter] = i_matRowIdx[l_nz];
      l_subNzCounter++;
    }
  }

  for ( unsigned int l_cc = std::min(l_nCols, i_nSubCols); l_cc < i_nSubCols + 1; l_cc++ ) {
    l_tmpMatColPtr[l_cc] = l_subNzCounter;
  }

  l_tmpMatVal.resize(l_subNzCounter);
  l_tmpMatRowIdx.resize(l_subNzCounter);

  o_subMatVal = l_tmpMatVal;
  o_subMatColPtr = l_tmpMatColPtr;
  o_subMatRowIdx = l_tmpMatRowIdx;

  return 0;
};

#endif

