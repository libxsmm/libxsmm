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



#include <map>
#include <string>

#include "proto/gxm.pb.h"
#include "io.hpp"

using namespace std;
using namespace gxm;

bool parseMachineConfig(const string fname, MachineParameter* param)
{
  bool success = ReadProtoFromText(fname, param);
  if(!success)
    printf("Failed to parse Machine Parameter file %s\n",fname.c_str());
  return success;
}

bool parseMLConfig(const string fname, NTGParameter* param)
{
  bool success = ReadProtoFromText(fname, param);
  if(!success)
    printf("Failed to parse ML Parameter file %s\n",fname.c_str());
  return success;
}

bool parseSolverConfig(const string fname, SolverParameter* param)
{
  bool success = ReadProtoFromText(fname, param);
  if(!success)
    printf("Failed to parse ML Parameter file %s\n",fname.c_str());
  return success;
}

