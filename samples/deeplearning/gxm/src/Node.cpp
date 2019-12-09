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
#if 0
    Task *sTask = this->getBasicTask(BASIC_TASK_SOLVE);
#endif

    string s = dynamic_cast<NNNode*>(fTask->getNode())->nname_;

#if 0
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
#endif

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

