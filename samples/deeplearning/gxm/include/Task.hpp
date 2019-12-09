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
#include <vector>
#include "MLNode.hpp"
#include <algorithm>
#include "Tensor.hpp"

#define BASIC_TASK_FORW   0
#define BASIC_TASK_BACK   1
#define BASIC_TASK_WGRAD  2
#define BASIC_TASK_SOLVE  3
#define CUSTOM_TASK_START 100

using namespace std;
using namespace gxm;

class Task
{
  protected:
    MLNode *node_;
    int taskId_;
    int basicTaskId_;
    int minBin_, maxBin_;

    vector<Task*> inputs_;
    vector<Task*> outputs_;
    vector<Task*> subTasks_;
    Task *parent_;

  public:
    Task(MLNode* n, int taskId, int basicTaskId)
    {
      this->node_ = n;
      this->taskId_ = taskId;
      this->basicTaskId_ = basicTaskId;
      this->minBin_ = 0;
      this->maxBin_ = 0;
      parent_ = NULL;
    }

    virtual ~Task(void) {}

    Task *createSubTask(int taskId) {
      Task *subTask = new Task(this->node_, taskId, basicTaskId_);
      this->subTasks_.push_back(subTask);
      subTask->parent_ = this;
      return subTask;
    }

    bool addForwDep(Task *dest) {
      if(dest == NULL) return false;
      // add only if task is not in the list
      if(std::find(outputs_.begin(), outputs_.end(), dest) == outputs_.end())
      {
        this->outputs_.push_back(dest);
        if(std::find(dest->inputs_.begin(), dest->inputs_.end(), this) == dest->inputs_.end())
          dest->inputs_.push_back(this);
        return true;
      }
      else
        return false;
    }

    bool addBackDep(Task *src) {
      if(src == NULL) return false;
      // add only if task is not in the list
      if(std::find(inputs_.begin(), inputs_.end(), src) == inputs_.end())
      {
        this->inputs_.push_back(src);
        if(std::find(src->outputs_.begin(), src->outputs_.end(), this) == src->outputs_.end())
          src->outputs_.push_back(this);
        return true;
      }
      else
        return false;
    }

    vector<Task*>& getForwDepTasks() { return this->outputs_; }
    vector<Task*>& getBackDepTasks() { return this->inputs_; }

    void setMinBin(int bin) { minBin_ = bin; }
    void setMaxBin(int bin) { maxBin_ = bin; }
    int getMinBin() { return minBin_; }
    int getMaxBin() { return maxBin_; }

    int getBasicTaskId() {return basicTaskId_; }
    int getTaskId() {return taskId_; }

    MLNode* getNode() { return node_; }

    void invoke() { node_->executeTask(basicTaskId_); }
    inline int numInputs() { return inputs_.size(); }
    inline int numOutputs() { return outputs_.size(); }
};
