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
#include <assert.h>
#include <string>
#include "MLNode.fwd.hpp"
#include "Shape.h"

using namespace std;

enum TensorDataType {DT_FLOAT, DT_INT, DT_BF16, DT_INT16, DT_DFP8, DT_INT8};
enum TensorBufType {DATA, DIFF, HISTORY, PRIVATE}; //also used as indices into tBuf_; should change
enum TensorType {INPUT, LABEL, ACT, INPUT_ACT, ACT_LABEL, CONVWEIGHT, CONVBIAS, FCWEIGHT, FCBIAS, BNORMSCALE, BNORMSHIFT, BNORMMEAN, BNORMVAR};
enum TensorLayoutType {NCHW, NHWC, NCHWV, KCRS, RSCK, LIBXSMM_CUSTOM_LAYOUT, NUM_LAYOUTS};

class Tensor;

class TensorBuf {
  protected:
    Tensor *tensor_;
    void *buf_; // Pointer to buffer
    void *lpbuf_; // Pointer to LP object
    void *prv_buf_;
    void *lp_prv_buf_;
    void **bufptr_;
    void **lpbufptr_;
    TensorLayoutType layout_type_;
    void *layout_;
    int offset_;
    int dType_; // Data type for this buffer
    int bType_; // Type of buffer (DATA/DIFF/HISTORY)
    long long int size_; // Size of this buffer
    int bin_; // Bin number assigned to this buffer
  public:
    TensorBuf(Tensor* tensor, int dtype = DT_FLOAT, int size = 0) : tensor_(tensor) {
      buf_ = NULL;
      lpbuf_ = NULL;
      prv_buf_ = NULL;
      lp_prv_buf_ = NULL;
      bufptr_ = NULL;
      lpbufptr_ = NULL;
      layout_type_ = NCHW;
      layout_ = NULL;
      offset_ = 0;
      dType_ = dtype;
      size_ = size;
      bin_ = 0;
    }

    Tensor* getTensor() { return tensor_; }

    void setBin(int bin) { bin_ = bin; }

    void setDataType(int t) { dType_ = t; }
    int getDataType() { return dType_; }

    void setBufferType(int t) { bType_ = t; }
    int getBufferType() { return bType_; }

    void setBufferSize(long long int size) { size_ = size; }
    long long int getBufferSize() { return size_; }

    void setOffset(int offset) { offset_ = offset; }
    int getOffset() { return offset_; }

    int getBin() { return bin_; }

    void setBuffer(void* bptr) { buf_ = bptr; }
    void* getBuffer() { return buf_; }

    void setLPBuffer(void* bptr) { lpbuf_ = bptr; }
    void* getLPBuffer() { return lpbuf_; }

    void setPrivBuffer(void* bptr) { prv_buf_ = bptr; }
    void* getPrivBuffer() { return prv_buf_; }

    void setLPPrivBuffer(void* bptr) { lp_prv_buf_ = bptr; }
    void* getLPPrivBuffer() { return lp_prv_buf_; }

    void setBufferPtr(void** bptr) { bufptr_ = bptr; }
    void** getBufferPtr() { return bufptr_; }

    void setLPBufferPtr(void** bptr) { lpbufptr_ = bptr; }
    void** getLPBufferPtr() { return lpbufptr_; }

    void setLayoutType(TensorLayoutType lt) { layout_type_ = lt; }
    TensorLayoutType getLayoutType() { return layout_type_; }

    void setLayout(void *layptr) { layout_ = layptr; }
    void* getLayout() { return layout_; }
};


class Tensor
{
  protected:
    string name_;
    Shape shape_; // Base logical shape of this tensor
    vector<TensorBuf*> tBuf_; // Structure holding pointer to buffer, its size, type and bin
    TensorType tType_; // Type of this tensor (Activation, Weight etc)
    MLNode *owner_;
    void *layout_; // Layout for this tensor (applies to all buffers)
    TensorLayoutType layout_type_;  // Layout type for this tensor (applies to all buffers)

  public:
    Tensor(string name)
    {
      this->name_ = name;
      tBuf_.push_back(new TensorBuf(this)); // Assume that tBuf_[0] is always the foward pass buffer
      layout_ = NULL;
      layout_type_ = NCHW;
    }

    virtual ~Tensor(void) {}

    MLNode *getOwner() { return owner_; }

    void setOwner(MLNode *owner) { owner_ = owner; }


    TensorBuf *addBuf(int dtype = DT_FLOAT, int size = 0)
    {
      TensorBuf *tb = new TensorBuf(this, dtype, size);
      this->tBuf_.push_back(tb);
      return tb;
    }

    void setShape(Shape* shape)
    {
      assert(shape->ndims <= MAX_DIMS);
      shape_.ndims = shape->ndims;
      for(int i=0; i<shape->ndims; i++)
        shape_.dims[i] = shape->dims[i];
      for(int i=shape->ndims; i<MAX_DIMS; i++)
        shape_.dims[i] = 0;
    }

    Shape *getShape() { return &shape_; }

    // Act, wt, shared, private, generic
    void setType(TensorType tt) { tType_ = tt; }
    int getType() { return tType_; }

    // float=1, int=2... create an enum for this
    void setBufDataType(int bufId, int tdt) { tBuf_[bufId]->setDataType(tdt); }
    int getBufDataType(int bufId) { return this->tBuf_[bufId]->getDataType(); }

    string getTensorName() { return name_; }

    int getNumDataBuffers() { return tBuf_.size(); }
    TensorBuf *getBuf(int bufId) { if(bufId < tBuf_.size()) return this->tBuf_[bufId]; else return NULL; }

    void setDataBuffer(int bufId, void* ptr) { this->tBuf_[bufId]->setBuffer(ptr); }
    void* getDataBuffer(int bufId) { return this->tBuf_[bufId]->getBuffer(); }

    void setDataBufferSize(int bufId, long long int size) { this->tBuf_[bufId]->setBufferSize(size); }
    long long int getDataBufferSize(int bufId) { return this->tBuf_[bufId]->getBufferSize(); }

    int getBufBin(int bufId) { return this->tBuf_[bufId]->getBin(); }
};

