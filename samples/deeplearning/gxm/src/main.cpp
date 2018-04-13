/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
#include "Eltwise.hpp"
#include "TypeList.hpp"


using namespace std;
using namespace gxm;

TypeList nodeTypes[] = {
  {"Accuracy", parseAccuracyParams, CreateMLNode<AccuracyNode,AccuracyParams>},
  {"FusedBatchNorm", parseFusedBNormParams, CreateMLNode<FusedBNormNode,FusedBNormParams>},
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
    string solvercfg = string();
    engine->create(TEST, mlcfg, solvercfg);
    engine->run(TEST);
  }
  return 0;
}

