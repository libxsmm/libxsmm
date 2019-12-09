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
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include "proto/gxm.pb.h"

using namespace std;
using namespace gxm;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromText(string, Message*);
bool ReadProtoFromBinary(string, Message*);
void WriteProtoToText(const Message&, string);
void ReadNWriteMeanFile(string, Message*, string);
void initSeeds(unsigned int*, int);
void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);
bool DecodeDatum(Datum* datum, bool is_color);
bool DecodeDatumNative(Datum* datum);
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);
cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
