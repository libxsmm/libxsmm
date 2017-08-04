# Image Convolution

This code sample aims to provide a simple piece of code, which takes an image and produces a visual result as well. For the convolution, LIBXSMM's DNN-domain is exercised. However, this code sample is not superceeding the collection of [DNN code samples](https://github.com/hfp/libxsmm/tree/master/samples/dnn)!

The executable can run with the following arguments (all arguments are optional):

> iconv   [\<filename-in\>  [\<nrepeat\>  [\<kw\>  [\<kh\>]  [\<filename-out\>]]]]

For stable timing (benchmark), the key operation (convolution) may be repeated (`nrepeat`). Further, `kw` and `kh` can specify the kernel-size of the convolution. The `filename-in` and `filename-out` name MHD-files (see [Meta Image File I/O](https://github.com/hfp/libxsmm/blob/master/documentation/libxsmm_aux.md#meta-image-file-io)) used as input and output respectively. The `filename-in` may not exist, and specify the image resolution (`w`[x`h`]; the file `wxh.mhd` is generated in this case).

