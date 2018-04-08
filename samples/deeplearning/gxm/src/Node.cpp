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
#include <vector>
#include "Node.hpp"

using namespace std;
using namespace gxm;

bool addedFD, addedBD;

void NNNode::createNNGraph(int mode)
{
  if(mode == TRAIN)
  {
    Task *fTask = this->getBasicTask(BASIC_TASK_FORW);
    Task *bTask = this->getBasicTask(BASIC_TASK_BACK);
    Task *wTask = this->getBasicTask(BASIC_TASK_WGRAD);
    Task *sTask = this->getBasicTask(BASIC_TASK_SOLVE);

    string s = dynamic_cast<NNNode*>(fTask->getNode())->nname_;

    if(wTask != NULL)
    {
      addedFD = wTask->addForwDep(sTask);
//      addedBD = fTask->addBackDep(sTask);
#ifdef DEBUG
      if(addedFD)
        printf("solver task (node %s) %p depends on weight task (node %s) %p\n", s.c_str(), sTask, s.c_str(), wTask);
      if(addedBD)
        printf("forward task (node %s) %p depends on solver task (node %s) %p\n",s.c_str(), fTask, s.c_str(), sTask);
#endif
    }

    for(auto it=nextNodes_.begin(); it != nextNodes_.end(); it++)
    {
      NNNode *nNode = *it;
      Task *fnTask = nNode->getBasicTask(BASIC_TASK_FORW);
      Task *bnTask = nNode->getBasicTask(BASIC_TASK_BACK);

      if(fnTask != NULL)
        addedFD = fTask->addForwDep(fnTask);
      if(bTask != NULL && bnTask != NULL) addedBD = bTask->addBackDep(bnTask);
      if(wTask != NULL && bnTask != NULL) addedBD = wTask->addBackDep(bnTask);
#ifdef DEBUG
      if(addedFD)
        printf("forward task (node %s) %p depends on forward task (node %s) %p\n",nNode->nname_.c_str(), fnTask, s.c_str(), fTask);
      if(bTask != NULL && bnTask != NULL)
        if(addedBD)
          printf("backward task (node %s) %p depends on backward task (node %s) %p\n",s.c_str(), bTask, nNode->nname_.c_str(), bnTask);
      if(wTask != NULL && bnTask != NULL)
        if(addedBD)
          printf("weight task (node %s) %p depends on backward task (node %s) %p\n", s.c_str(), wTask, nNode->nname_.c_str(), bnTask);
#endif
      nNode->createNNGraph(mode);
    }

    // Handle last node
    if(nextNodes_.size() == 0)
    {
      if(bTask != NULL)
      {
        addedFD = fTask->addForwDep(bTask);
#ifdef DEBUG
        if(addedFD)
          printf("backward task (node %s) %p depends on forward task (node %s) %p\n",s.c_str(), bTask, s.c_str(), fTask);
#endif
      }
    }
  }
  else if(mode == TEST)
  {
    Task *fTask = this->getBasicTask(BASIC_TASK_FORW);

    for(auto it=nextNodes_.begin(); it != nextNodes_.end(); it++)
    {
      NNNode *nNode = *it;
      Task *fnTask = nNode->getBasicTask(BASIC_TASK_FORW);
      fTask->addForwDep(fnTask);
#ifdef DEBUG
      printf("forward task %p depends on forward task %p\n",fnTask,fTask);
#endif
      nNode->createNNGraph(mode);
    }
  }
}

