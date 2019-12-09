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
#include <vector>
#include <algorithm>
#include <list>
#include <algorithm>
#include "Params.hpp"
#include "MLNode.hpp"
#include "Engine.fwd.hpp"
#include "Task.hpp"
#include "proto/gxm.pb.h"

using namespace std;
using namespace gxm;
#ifdef USE_MLSL
#include "mlsl.hpp"
#endif

class NNParams : public MLParams
{
  protected:
    vector<string> top_;
    vector<string> bottom_;
    string nname_;
    string type_;
    int mode_;
    bool bp_flag_;

  public:
    NNParams(void) {}
    virtual ~NNParams(void) {}

    void set_top_names(string name) { top_.push_back(name); }
    void set_bottom_names(string name) { bottom_.push_back(name); }
    void set_node_name(string nname) { nname_ = nname; }
    void set_node_type(string type) {type_ = type; }
    void set_mode(int mode) { mode_ = mode; }
    void set_bprop_flag(bool flag) { bp_flag_ = flag; }

    string get_node_name() { return nname_; }
    vector<string>& get_top_names() { return top_; }
    vector<string>& get_bottom_names() { return bottom_; }
    string get_node_type() { return type_; }
    int get_mode() { return mode_; }
    bool get_bprop_flag() { return bp_flag_; }
};

class NNNode : public MLNode
{
  public:
    NNNode(NNParams* p, MLEngine* e) : MLNode(p, e)
    {
      for(int i = 0; i < 4; i++) tBasic_[i] = NULL;
    }

    virtual ~NNNode(void)
    {
      for(int i = 0; i < 4; i++) if(tBasic_[i] != NULL) { delete tBasic_[i]; tBasic_[i] = NULL; }
    }

    void createTasks(list<Task*>, int) {}
    virtual void createStrategy(int) {}

    virtual void forwardPropagate() {}
    virtual void backPropagate() {}
    virtual void weightUpdate() {}
    virtual void solverStep() {}

    int executeTask(int taskId)
    {
      if(taskId == 0)
      {
        forwardPropagate();
      }
      else if(taskId == 1)
      {
        backPropagate();
      }
      else if(taskId == 2)
      {
        weightUpdate();
      }
      else if(taskId == 3)
      {
        solverStep();
      }
      return 0;
    }

    void enqueTask(int pos) {}

    virtual void createPersistentTask() {}

    void setNextNode(NNNode* next)
    {
      //check if next is already in the nextNodes list
      if(std::find(nextNodes_.begin(), nextNodes_.end(), next) == nextNodes_.end())
      {
        nextNodes_.push_back(next);
        next->prevNodes_.push_back(this);
      }
    }

    void setPrevNode(NNNode* prev)
    {
      //check if prev is already in the prevNodes list
      if(std::find(prevNodes_.begin(), prevNodes_.end(), prev) == prevNodes_.end())
      {
        prevNodes_.push_back(prev);
        prev->nextNodes_.push_back(this);
      }
    }

    Task *getBasicTask(int type)
    {
      int index = -1;
      if(type == 0 || (type == 1 && bp_flag_) || (type > 1 && has_weights_))
        index = type;
      if(index != -1) {
        if(tBasic_[index] == NULL) tBasic_[index] = new Task(this, -1, type);
        return tBasic_[index];
      }
      return NULL;
    }

    void createNNGraph(int mode);

    void setNodeType(string type) { ntype_ = type; }

    string getNodeType() { return ntype_; }
    string getNodeName() { return nname_; }
    int getMode() { return mode_; }

    int getNumPrevNodes() { return prevNodes_.size(); }
    int getNumNextNodes() { return nextNodes_.size(); }

    NNNode* getPrevNode(int i) { if(prevNodes_.size() > 0) return prevNodes_[i]; else return NULL; }
    NNNode* getNextNode(int i) { if(nextNodes_.size() > 0) return nextNodes_[i]; else return NULL; }

    int get_num_tops() { return top_.size(); }
    void set_top_compute_engine(int e) { top_compute_engine_ = e; }
    int get_bot_compute_engine() { return bot_compute_engine_; }
    void set_next_node_type(string s) {next_ntype_ = s;}

    void refineTask(){}

    virtual void createCheckPoint() {}
    virtual void restoreCheckPoint() {}

  protected:
    string nname_, ntype_, next_ntype_;
    vector<string> top_;
    vector<string> bottom_;
    int mode_;
    bool bp_flag_;
    bool has_weights_;
    vector<NNNode*> prevNodes_;
    vector<NNNode*> nextNodes_;
    int top_compute_engine_, bot_compute_engine_;
#ifdef USE_MLSL
    MLSL::Operation* op_;
#endif


    // 0-Forw, 1-Back, 2-WGrad, 3-Solver
    Task *tBasic_[4];
};

