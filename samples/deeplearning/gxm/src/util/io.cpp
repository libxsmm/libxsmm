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
#include "io.hpp"

using namespace std;
using namespace gxm;

const int kProtoReadBytesLimit = INT_MAX;


bool ReadProtoFromText(string fname, Message* proto)
{
  int fm = open(fname.c_str(), O_RDONLY);
  if (fm == -1)
  {
    printf("File %s not found\n",fname.c_str());
    return false;
  }
  FileInputStream* input = new FileInputStream(fm);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fm);
  return success;
}

bool ReadProtoFromBinary(string fname, Message* proto)
{
  int fd = open(fname.c_str(), O_RDONLY);
  if (fd == -1)
  {
    printf("File %s not found\n",fname.c_str());
    return false;
  }

  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  if(!coded_input->ConsumedEntireMessage())
  {
    printf("parsing failed\n");
    exit(-1);
  }

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToText(const Message& proto, string filename)
{
  int fd = open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  google::protobuf::TextFormat::Print(proto, output);
  delete output;
  close(fd);
}

void ReadNWriteMeanFile(string fname, Message* proto, string outname)
{
  ReadProtoFromBinary(fname, proto);
  WriteProtoToText(*proto, outname);
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    printf("Could not decode datum\n");
  }
  return cv_img;
}

bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
      CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    printf("Could not decode datum\n");
  }
  return cv_img;
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}
#endif

void initSeeds(unsigned int *seeds, int nthreads)
{
    for(int i=0; i<nthreads*16; i++)
    {
      if(i%16 == 0)
        seeds[i] = rand();
      else
        seeds[i] = 0;
    }
}
