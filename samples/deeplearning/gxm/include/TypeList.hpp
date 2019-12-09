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



typedef MLParams *(*ParseFunc)(NodeParameter* np);
typedef MLNode *(*CreateFunc)(MLParams* p, MLEngine* e);

typedef struct TypeList_{
  std::string typeName;
  ParseFunc parse;
  CreateFunc create;
} TypeList;

extern TypeList nodeTypes[];
extern const int numTypes;


