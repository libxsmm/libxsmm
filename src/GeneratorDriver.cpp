/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

/**
 * @file
 * This file is part of GemmCodeGenerator.
 *
 * @author Alexander Heinecke (alexander.heinecke AT mytum.de, http://www5.in.tum.de/wiki/index.php/Alexander_Heinecke,_M.Sc.,_M.Sc._with_honors)
 *
 * @section LICENSE
 * Copyright (c) 2012-2014, Technische Universitaet Muenchen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE  USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 * <DESCRIPTION>
 */

#include <iostream>
#include <cstdlib>
#include <fstream>

#include "GeneratorCSC.hpp"
#include "GeneratorDense.hpp"

void generate_custom_signature_right(std::ofstream& out, std::string tRoutineName, bool bSP, std::string tPrefetch) {
  if (bSP == true) {
    if (tPrefetch.compare("nopf") == 0) {
      out << "void " << tRoutineName << "(const float* A, const float* values, float* C)" << std::endl << "{" << std::endl;
    } else {
      out << "void " << tRoutineName << "(const float* A, const float* values, float* C, const float* A_prefetch = NULL, const float* B_prefetch = NULL, const float* C_prefetch = NULL)" << std::endl << "{" << std::endl;
    }
  } else {
    if (tPrefetch.compare("nopf") == 0) {
      out << "void " << tRoutineName << "(const double* A, const double* values, double* C)" << std::endl << "{" << std::endl;
    } else {
      out << "void " << tRoutineName << "(const double* A, const double* values, double* C, const double* A_prefetch = NULL, const double* B_prefetch = NULL, const double* C_prefetch = NULL)" << std::endl << "{" << std::endl;
    }
  }
}

void generate_custom_signature_left(std::ofstream& out, std::string tRoutineName, bool bSP, std::string tPrefetch) {
  if (bSP == true) {
    if (tPrefetch.compare("nopf") == 0) {
      out << "void " << tRoutineName << "(const float* values, const float* B, float* C)" << std::endl << "{" << std::endl;
    } else {
      out << "void " << tRoutineName << "(const float* values, const float* B, float* C, const float* A_prefetch = NULL, const float* B_prefetch = NULL, const float* C_prefetch = NULL)" << std::endl << "{" << std::endl;
    }
  } else {
    if (tPrefetch.compare("nopf") == 0) {
      out << "void " << tRoutineName << "(const double* values, const double* B, double* C)" << std::endl << "{" << std::endl;
    } else {
      out << "void " << tRoutineName << "(const double* values, const double* B, double* C, const double* A_prefetch = NULL, const double* B_prefetch = NULL, const double* C_prefetch = NULL)" << std::endl << "{" << std::endl;
    }
  }
}

void generate_custom_signature_dense(std::ofstream& out, std::string tRoutineName, bool bSP, std::string tPrefetch) {
  if (bSP == true) {
    if (tPrefetch.compare("nopf") == 0) {
      out << "void " << tRoutineName << "(const float* A, const float* B, float* C)" << std::endl << "{" << std::endl;
    } else {
      out << "void " << tRoutineName << "(const float* A, const float* B, float* C, const float* A_prefetch = NULL, const float* B_prefetch = NULL, const float* C_prefetch = NULL)" << std::endl << "{" << std::endl;
    }
  } else {
    if (tPrefetch.compare("nopf") == 0) {
      out << "void " << tRoutineName << "(const double* A, const double* B, double* C)" << std::endl << "{" << std::endl;
    } else {
      out << "void " << tRoutineName << "(const double* A, const double* B, double* C, const double* A_prefetch = NULL, const double* B_prefetch = NULL, const double* C_prefetch = NULL)" << std::endl << "{" << std::endl;
    }
  }
}

void generate_epilogue(std::ofstream& out) {
  out << "}" << std::endl << std::endl;
}

void generator_seissol_sparse(std::string tFileOut, std::string tRoutineName, std::string tFileIn, int nM, int nN, int nK, int nDenseLDA, int nDenseLDB, int nDenseLDC, bool bAdd, bool bSP, std::string tVec, std::string tPrefetch) {
  std::string generated_code;
  std::ofstream out;

  out.open(tFileOut.c_str(), std::ios::app);

  seissolgen::GeneratorCSC* gen;

  gen = new seissolgen::GeneratorCSC(bAdd, bSP, tVec);

  if (nDenseLDB < 1) {
    generate_custom_signature_right(out, tRoutineName, bSP, tPrefetch);
    generated_code = gen->generate_code_right(tFileIn, nM, nN, nK, nDenseLDA, nDenseLDC);
    out << generated_code;
  } else if (nDenseLDA < 1) {
    generate_custom_signature_left(out, tRoutineName, bSP, tPrefetch);
    generated_code = gen->generate_code_left(tFileIn, nM, nN, nK, nDenseLDB, nDenseLDC);
    out << generated_code;
  }

#ifdef DEBUG
  std::cout << "code was generated and exported to " << tFileOut << ":" << std::endl;
  //  std::cout << generated_code << std::endl << std::endl;
#endif
  generate_epilogue(out);
  out.close();

  delete gen;
}

void generator_seissol_dense(std::string tFileOut, std::string tRoutineName, int nDenseM, int nDenseN, int nDenseK, int nDenseLDA, int nDenseLDB, int nDenseLDC, bool bAdd, std::string tVec, std::string tPrefetch, bool bSP) {
  std::string generated_code;
  std::ofstream out;
  seissolgen::GeneratorDense* gen;

  gen = new seissolgen::GeneratorDense(bAdd, tVec, tPrefetch, bSP);

  out.open(tFileOut.c_str(), std::ios::app);
  // generate code
  generated_code = gen->generate_dense(true, nDenseM, nDenseN, nDenseK, nDenseLDA, nDenseLDB, nDenseLDC);

  // generate code
  generate_custom_signature_dense(out, tRoutineName, bSP, tPrefetch);

  // write code to file
  out << generated_code;

  // close function
  generate_epilogue(out);

  out.close();

#ifdef DEBUG
  std::cout << "code was generated and exported to " << tFileOut << ":" << std::endl;
  //  std::cout << generated_code << std::endl << std::endl;
#endif

  delete gen;
}

void print_help() {
  std::cout << std::endl << "wrong usage -> exit!" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "Usage (sparse*dense=dense, dense*sparse=dense):" << std::endl;
  std::cout << "    sparse" << std::endl;
  std::cout << "    filename to append" << std::endl;
  std::cout << "    rountine name" << std::endl;
  std::cout << "    matrix input" << std::endl;
  std::cout << "    M" << std::endl;
  std::cout << "    N" << std::endl;
  std::cout << "    K" << std::endl;
  std::cout << "    LDA" << std::endl;
  std::cout << "    LDB" << std::endl;
  std::cout << "    LDC" << std::endl;
  std::cout << "    0: no add, otherwise: add" << std::endl;
  std::cout << "    ARCH: noarch, wsm, snb, hsw, knc" << std::endl;
  std::cout << "    PREFETCH: none, pfsigonly, other dense options fall-back to pfsigonly" << std::endl;
  std::cout << "    PRECISION: SP, DP" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << "Usage (dense*dense=dense):" << std::endl;
  std::cout << "    dense" << std::endl;
  std::cout << "    filename to append" << std::endl;
  std::cout << "    rountine name" << std::endl;
  std::cout << "    M" << std::endl;
  std::cout << "    N" << std::endl;
  std::cout << "    K" << std::endl;
  std::cout << "    LDA" << std::endl;
  std::cout << "    LDB" << std::endl;
  std::cout << "    LDC" << std::endl;
  std::cout << "    0: no add, otherwise: add" << std::endl;
  std::cout << "    ARCH: noarch, wsm, snb, hsw, knc" << std::endl;
  std::cout << "    PREFETCH: none, pfsigonly, BL2viaC, AL2, curAL2, AL2jpst, AL2_BL2viaC, curAL2_BL2viaC, AL2jpst_BL2viaC" << std::endl;
  std::cout << "    PRECISION: SP, DP" << std::endl;
  std::cout << std::endl << std::endl;
  std::cout << std::endl << std::endl;
}

int main(int argc, char* argv []) {
  // check argument count for a valid range
  if (argc != 14 && argc != 15) {
    print_help();
    return -1;
  }

  std::string tFileOut;
  std::string tRoutineName;
  std::string tType;
  std::string tVec;
  std::string tPrefetch;
  std::string tPrecision;

  tType.assign(argv[1]);

  // some additional sparse/dense parameters checks
  if ( tType != "sparse" && tType != "dense" ) {
    print_help();
    return -1;
  }
  if ( (tType == "sparse" && argc != 15) || (tType == "dense" && argc != 14) ) {
    print_help();
    return -1;
  }

  if (tType == "sparse") {
    int nM = 0;
    int nN = 0;
    int nK = 0;
    int nDenseLDA = 0;
    int nDenseLDB = 0;
    int nDenseLDC = 0;
    int nAdd;
    std::string tFileIn;
    bool bAdd = true;
    bool bSP = false;

    tFileOut.assign(argv[2]);
    tRoutineName.assign(argv[3]);
    tFileIn.assign(argv[4]);
    nM = atoi(argv[5]);
    nN = atoi(argv[6]);
    nK = atoi(argv[7]);
    nDenseLDA = atoi(argv[8]);
    nDenseLDB = atoi(argv[9]);
    nDenseLDC = atoi(argv[10]);
    nAdd = atoi(argv[11]);
    tVec.assign(argv[12]);
    tPrefetch.assign(argv[13]);
    tPrecision.assign(argv[14]);

    if ( (tVec.compare("wsm") != 0)        && 
         (tVec.compare("snb") != 0)        && 
         (tVec.compare("hsw") != 0)        &&
         (tVec.compare("knc") != 0)        && 
         (tVec.compare("noarch") != 0) ) {
      print_help();
      return -1;
    }

    if ( (tPrefetch.compare("nopf") != 0 ) &&
         (tPrefetch.compare("pfsigonly") != 0 ) &&
         (tPrefetch.compare("BL2viaC") != 0 ) &&
         (tPrefetch.compare("curAL2") != 0 ) &&
         (tPrefetch.compare("curAL2_BL2viaC") != 0 ) &&
         (tPrefetch.compare("AL2") != 0 ) &&
         (tPrefetch.compare("AL2_BL2viaC") != 0 ) &&
         (tPrefetch.compare("AL2jpst") !=0 ) &&
         (tPrefetch.compare("AL2jpst_BL2viaC") !=0 ) ) {
      print_help();
      return -1;
    }

    if (nAdd == 0)
      bAdd = false;

    if ( tPrecision.compare("SP") == 0 ) {
      bSP = true;
    } else if ( tPrecision.compare("DP") == 0 ) {
      bSP = false;
    } else {
      print_help();
      return -1;
    }

    if (nDenseLDA < 1 && nDenseLDB < 1) {
      print_help();
      return -1;
    }

    if (nDenseLDC < 1) {
      print_help();
      return -1;
    }

    generator_seissol_sparse(tFileOut, tRoutineName, tFileIn, nM, nN, nK, nDenseLDA, nDenseLDB, nDenseLDC, bAdd, bSP, tVec, tPrefetch);
  }

  if (tType == "dense") {
    int nDenseM = 0;
    int nDenseN = 0;
    int nDenseK = 0;
    int nDenseLDA = 0;
    int nDenseLDB = 0;
    int nDenseLDC = 0;
    int nAdd = 0;

    bool bAdd = true;
    bool bSP = false;

    tFileOut.assign(argv[2]);
    tRoutineName.assign(argv[3]);
    nDenseM = atoi(argv[4]);
    nDenseN = atoi(argv[5]);
    nDenseK = atoi(argv[6]);
    nDenseLDA = atoi(argv[7]);
    nDenseLDB = atoi(argv[8]);
    nDenseLDC = atoi(argv[9]);
    nAdd = atoi(argv[10]);
    tVec.assign(argv[11]);
    tPrefetch.assign(argv[12]);
    tPrecision.assign(argv[13]);

    if ( (tVec.compare("wsm") != 0)        && 
         (tVec.compare("snb") != 0)        && 
         (tVec.compare("hsw") != 0)        &&
         (tVec.compare("knc") != 0)        && 
         (tVec.compare("noarch") != 0) ) {
      print_help();
      return -1;
    }

    if ( (tPrefetch.compare("nopf") != 0 ) &&
         (tPrefetch.compare("pfsigonly") != 0 ) &&
         (tPrefetch.compare("BL2viaC") !=0 ) && 
         (tPrefetch.compare("curAL2") != 0 ) &&
         (tPrefetch.compare("curAL2_BL2viaC") !=0 ) && 
         (tPrefetch.compare("AL2") != 0 ) &&
         (tPrefetch.compare("AL2_BL2viaC") !=0 ) && 
         (tPrefetch.compare("AL2jpst") !=0 ) &&
         (tPrefetch.compare("AL2jpst_BL2viaC") !=0 ) ) {
      print_help();
      return -1;
    }

    if ( tPrecision.compare("SP") == 0 ) {
      bSP = true;
    } else if ( tPrecision.compare("DP") == 0 ) {
      bSP = false;
    } else {
      print_help();
      return -1;
    }

    if (nAdd == 0)
      bAdd = false;

    generator_seissol_dense(tFileOut, tRoutineName, nDenseM, nDenseN, nDenseK, nDenseLDA, nDenseLDB, nDenseLDC, bAdd, tVec, tPrefetch, bSP);
  }
}

