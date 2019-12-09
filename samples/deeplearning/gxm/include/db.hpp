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



#ifndef _DB_HPP_
#define _DB_HPP_

#include <string>
#include <utility>

using namespace std;

class Cursor {
 public:
  Cursor() { }
  virtual ~Cursor() { }
  virtual void SeekToFirst() = 0;
  virtual void Next(int skip = 0) = 0;
  virtual string key() = 0;
  virtual string value() = 0;
  virtual int count() = 0;
  virtual std::pair<void*, size_t> valuePointer() = 0;
  virtual bool valid() = 0;

  //DISABLE_COPY_AND_ASSIGN(Cursor);
};

class DB {
 public:
  DB() { }
  virtual ~DB() { }
  virtual void Open(const string& source)= 0;
  virtual void Close() = 0;
  virtual Cursor* NewCursor() = 0;

  //DISABLE_COPY_AND_ASSIGN(DB);
};

//DB* GetDB(DataParameter::DB backend);
DB* GetDB(const string& backend);

#endif  // _DB_HPP_
