# Image Convolution

This code sample aims to provide a simple piece of code, which takes an image and produces a visual result as well. For the convolution, LIBXSMM's DNN-domain is exercised. However, this code sample is not superceeding the collection of [DNN code samples](https://github.com/hfp/libxsmm/tree/master/samples/dnn).

By default (Makefile: OMP=0), this code does not use multiple threads since parallelization is per number of image-channels and per multiple images. This sample code processes only a single image which consists of a single channel (eventually multiple times as per `nrepeat`).

The executable can run with the following arguments (all arguments are optional):

> iconv   [\<filename-in\>  [\<nrepeat\>  [\<kw\>  [\<kh\>]  [\<filename-out\>]]]]

For stable timing (benchmark), the key operation (convolution) may be repeated (`nrepeat`). Further, `kw` and `kh` can specify the kernel-size of the convolution. The `filename-in` and `filename-out` name MHD-files (see [Meta Image File I/O](https://github.com/hfp/libxsmm/blob/master/documentation/libxsmm_aux.md#meta-image-file-io)) used as input and output respectively. The `filename-in` may not exist, and specify the image resolution (`w`[x`h`] where the file `wxh.mhd` is generated in this case).

To load an image from a common format (JPG, PNG, etc.), one may save the raw data using for instance [IrfanView](http://www.irfanview.com/) and rely on a "header-only" MHD-file (plain text). This may look like:

```
NDims = 2
DimSize = 202 134
ElementType = MET_UCHAR
ElementNumberOfChannels = 1
ElementDataFile = mhd_image.raw
```

In the above case, a single channel (gray-scale) 202x134-image is described (actual pixel data is stored in `mhd_image.raw`). The pixel type is according to the `libxsmm_mhd_elemtype` ([libxsmm_mhd.h](https://github.com/hfp/libxsmm/blob/master/include/libxsmm_mhd.h#L38)).

