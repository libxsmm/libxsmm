/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Sasikanth Avancha, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#pragma once

#include <string>
#include "proto/gxm.pb.h"
using namespace std;
using namespace gxm;

bool parseMachineConfig(const string mcFile, MachineParameter* param);
bool parseMLConfig(const string mlFile, NTGParameter* param);
//bool parseStrategyConfig(const string& strategyFile, StrategyParameter* param); // Read saved tunning parameters
bool parseSolverConfig(const string solverFile, SolverParameter* param);
