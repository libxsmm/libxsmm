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


#include <string>
#include "string.h"
#include "Engine.hpp"
#include "Node.hpp"
#include "Accuracy.hpp"
#include "Concat.hpp"
#include "DummyData.hpp"
#include "ImageData.hpp"
#include "JitterData.hpp"
#include "LMDBData.hpp"
#include "Conv.hpp"
#include "FullyConnected.hpp"
#include "ReLU.hpp"
#include "Dropout.hpp"
#include "Pooling.hpp"
#include "SoftmaxLoss.hpp"
#include "Split.hpp"
#include "FusedBNorm.hpp"
#include "FusedConvBN.hpp"
#include "Eltwise.hpp"
#include "TypeList.hpp"


using namespace std;
using namespace gxm;

TypeList nodeTypes[] = {
  {"Accuracy", parseAccuracyParams, CreateMLNode<AccuracyNode,AccuracyParams>},
  {"FusedBatchNorm", parseFusedBNormParams, CreateMLNode<FusedBNormNode,FusedBNormParams>},
  {"FusedConvBN", parseFusedConvBNParams, CreateMLNode<FusedConvBNNode,FusedConvBNParams>},
  {"Eltwise", parseEltwiseParams, CreateMLNode<EltwiseNode,EltwiseParams>},
  {"Split", parseSplitParams, CreateMLNode<SplitNode,SplitParams>},
  {"Concat", parseConcatParams, CreateMLNode<ConcatNode,ConcatParams>},
  {"DummyData", parseDummyDataParams, CreateMLNode<DummyDataNode,DummyDataParams>},
  {"ImageData", parseImageDataParams, CreateMLNode<ImageDataNode,ImageDataParams>},
  {"JitterData", parseJitterDataParams, CreateMLNode<JitterDataNode,JitterDataParams>},
  {"LMDBData", parseLMDBDataParams, CreateMLNode<LMDBDataNode,LMDBDataParams>},
  {"Convolution", parseConvParams, CreateMLNode<ConvNode,ConvParams>},
  {"FullyConnected", parseFCParams, CreateMLNode<FCNode,FCParams>},
  {"Pooling", parsePoolingParams, CreateMLNode<PoolingNode,PoolingParams>},
  {"ReLU", parseReLUParams, CreateMLNode<ReLUNode,ReLUParams>},
  {"Dropout", parseDropoutParams, CreateMLNode<DropoutNode,DropoutParams>},
  {"SoftmaxWithLoss", parseSoftmaxParams, CreateMLNode<SoftmaxLossNode,SoftmaxLossParams>}
};

const int numTypes = sizeof(nodeTypes)/sizeof(nodeTypes[0]);

int main(int argc, char* argv[])
{
  //Command-line arguments for MLConfig, SolverConfig, MachineConfig
#ifdef USE_MLSL
  MLSL::Environment::GetEnv().Init(&argc, &argv);
#endif
  // Create MLEngine instance
  MLEngine *engine = new MLEngine();

  if(strcmp(argv[1], "train") == 0)
  {
    string mlcfg(argv[2]);
    string solvercfg(argv[3]);

    engine->create(TRAIN, mlcfg, solvercfg);
    engine->run(TRAIN);
  }
  else if(strcmp(argv[1], "test") == 0)
  {
    string mlcfg(argv[2]);
    string solvercfg(argv[3]);
    engine->create(TEST, mlcfg, solvercfg);
    engine->run(TEST);
  }

#ifdef USE_MLSL
  MLSL::Environment::GetEnv().Finalize();
#endif

  return 0;
}

