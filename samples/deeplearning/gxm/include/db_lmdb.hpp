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


#ifdef USE_LMDB
#ifndef _LMDB_HPP_
#define _LMDB_HPP_

#include <string>
#include <utility>
#include <vector>

#include "lmdb.h"

#include "db.hpp"

inline void MDB_CHECK(int mdb_status) {
  //CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
  if(mdb_status != MDB_SUCCESS)
  {
    printf("MDB Error: %s\n",mdb_strerror(mdb_status));
    exit(1);
  }
}

class LMDBCursor : public Cursor {
 public:
  explicit LMDBCursor(MDB_txn* mdb_txn, MDB_cursor* mdb_cursor, int count)
    : mdb_txn_(mdb_txn), mdb_cursor_(mdb_cursor), valid_(false), count_(count) {
    SeekToFirst();
  }
  virtual ~LMDBCursor() {
    mdb_cursor_close(mdb_cursor_);
    mdb_txn_abort(mdb_txn_);
  }
  virtual void SeekToFirst() { Seek(MDB_FIRST); }
  virtual void Next(int skip = 0) { Seek(MDB_NEXT, skip); }
  virtual string key() {
    return string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  }
  virtual string value() {
    return string(static_cast<const char*>(mdb_value_.mv_data),
        mdb_value_.mv_size);
  }
  virtual int count() {
    return count_;
  }
  virtual std::pair<void*, size_t> valuePointer() {
    return std::make_pair(mdb_value_.mv_data, mdb_value_.mv_size);
  }

  virtual bool valid() { return valid_; }

 private:
  void Seek(MDB_cursor_op op, int skip = 0) {
    int mdb_status = MDB_SUCCESS;
    if(op == MDB_NEXT) for(int i = 0; i < skip; i++) {
      mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, 0, MDB_NEXT);
      if (mdb_status == MDB_NOTFOUND) {
        mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, 0, MDB_FIRST);
        //printf("LMDB wrap around\n");
      }
      if (mdb_status != MDB_SUCCESS) break;
    }
    if(mdb_status == MDB_SUCCESS)
      mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
    if (mdb_status == MDB_NOTFOUND) {
      mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST);
      //printf("LMDB wrap around\n");
      //valid_ = false;
    }
    MDB_CHECK(mdb_status);
    valid_ = true;
  }

  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
  bool valid_;
  int count_;
};

class LMDB : public DB {
 public:
  LMDB() : mdb_env_(NULL) { }
  virtual ~LMDB() { Close(); }
  virtual void Open(const string& sourc);
  virtual void Close() {
    if (mdb_env_ != NULL) {
      mdb_dbi_close(mdb_env_, mdb_dbi_);
      mdb_env_close(mdb_env_);
      mdb_env_ = NULL;
    }
  }
  virtual LMDBCursor* NewCursor();

 private:
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
};
#endif
#endif

