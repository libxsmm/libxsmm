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


#pragma once
#include <map>
#include <list>
#include <vector>
#include <algorithm>
#include <set>
#include <omp.h>
#include <sys/time.h>
#include "proto/gxm.pb.h"
#include "Engine.fwd.hpp"
#include "MLNode.fwd.hpp"
#include "Config.hpp"
#include "Task.hpp"
#include "Solver.hpp"
#include "libxsmm.h"

using namespace std;
using namespace gxm;

extern int iter;

#ifdef USE_MLSL
#include "mlsl.hpp"
//using namespace MLSL;
#endif

#define TRAIN 0
#define TEST 1
#define START_GUARD_BAND 64
#define END_GUARD_BAND 64
#define CANARY 0x7F
#define NDIFFS 10

struct dupChecker_ {
  inline dupChecker_() : tmpSet() {}
  inline bool operator()(Task *t) {
    return tmpSet.insert(t).second;
  }
  private:
    std::set<Task *> tmpSet;
};

class MLEngine
{
  protected:
    NTGParameter ntgparam_;
    NodeParameter np_;
    SolverParameter sparam_;
#ifdef USE_MLSL
    MLSL::Distribution *data_parallelism;
    MLSL::Session *session_;
#endif
    vector<MLNode*> ntg_;
    list<Task*> etg_[2]; // 0 - Training, 1 - Validation/testing
    SolverParams *solverParams_;
    SolverNode* solver_;
    Tensor* tenScratch_;

    struct TensorPair
    {
      string name;
      Tensor* t;
    };
    typedef list<TensorPair> TensorList;
    typedef TensorList::iterator Iter;
    typedef map<string, Iter> Tmap;

    Tmap inTensorMap_, outTensorMap_, weightTensorMap_, biasTensorMap_, statsTensorMap_;
    TensorList defTList_, inTList_, outTList_, wTList_, biasTList_, statsTList_;

    bool inferenceOnly_, load_from_checkpoint_;
    string checkpoint_dir_, checkpoint_format_;
    int num_epochs_, exec_mode_, current_epoch_, current_batch_;
    int data_type_;
    int num_machines_, num_machine_groups_, num_threads_;
    int batch_size_, num_train_batches_, num_test_batches_, num_test_views_;
    int global_node_id_;
    float lr_, *wt_lr_mult_, *wt_decay_mult_;
    float *bias_lr_mult_, *bias_decay_mult_;

    void *input_buf_=NULL;
    void *fact_buf_=NULL, *bact_buf_=NULL;
    void *weight_buf_=NULL, *wdiff_buf_=NULL, *winc_buf_=NULL;
    void *bias_buf_=NULL, *bidiff_buf_=NULL, *biinc_buf_=NULL, *stats_buf_=NULL;
    int total_weights_, total_biases_;

    vector<int> input_can_ptr;
    vector<int> fact_can_ptr, bact_can_ptr;
    vector<int> wt_can_ptr, wdiff_can_ptr, winc_can_ptr;
    vector<int> bias_can_ptr, stats_can_ptr, bidiff_can_ptr, biinc_can_ptr;
    int ic, fac, bac, wtc, wdc, wic, bic, sic, bidc, biic;

    void create_schedule(int);
    void optimize_schedule(int);
    void allocate_tensor_memory(Tensor*, int, void*);
    void clear_history(TensorList);
    int find_in_nodeTypeList(string);
    void checkpoint(TensorList L);
    void read_checkpoint_file(TensorBuf*, string, string);
    void load_checkpoint(TensorList, string);
    void canary_check(void*, vector<int>&, int);
    void* allocate_memory(string, TensorList, int, vector<int>&, int*, long long int*, long long int*, int);
    void* allocate_gradient_tensor(TensorList, int, int, long long int);
    void insertSplitNodes(NTGParameter& p, NTGParameter* ps);
    void quantize_and_transpose_weights(TensorList L);

  public:
    MLEngine() {}
    virtual ~MLEngine() {}

    void create(int mode, string ntgConfig, string solverConfig);
    bool register_tensor(string name, int type, Tensor* t);
    Tensor* get_tensor(string name, int type);
    void execute_on_thread(int num_threads, MLNode* node, void (*fname)(int tid));
    void set_global_strategy(MachineParameter* mparam);
    void run(int mode);

    SolverNode* getSolver() { return solver_; }
    TensorBuf* getScratchBuffer() { return tenScratch_->getBuf(DATA); }

    bool is_inference_only() { return inferenceOnly_; }

    int get_num_threads() { return num_threads_; }
    int get_num_machines() { return num_machines_; }
    int get_num_machine_groups() { return num_machine_groups_; }
    int get_num_epochs() { return num_epochs_;}
    int get_current_epoch() { return current_epoch_; }
    int get_current_batch() { return current_batch_; }
    int get_execution_mode() { return exec_mode_; }
    int get_global_node_id() { return global_node_id_; }
    int get_num_train_batches() { return num_train_batches_; }
    int get_num_test_batches() { return num_test_batches_; }
    int get_num_test_views() {return num_test_views_; }
    int get_batch_size() { return batch_size_; }

    void set_batch_size(int b) {batch_size_ = b; }
    void set_num_train_batches(int ntrainb) {num_train_batches_ = ntrainb; }
    void set_num_test_batches(int ntestb) {num_test_batches_ = ntestb; }
    void set_num_test_views(int ntestv) {num_test_views_ = ntestv; }
    void set_learning_rate(float lr) { lr_ = lr; }
#ifdef USE_MLSL
    MLSL::Distribution* get_distribution() { return data_parallelism; }
    MLSL::Session *get_session() { return session_; }
#endif

};

