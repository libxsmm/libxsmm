# MHD Image I/O

This code sample aims to provide a simple piece of code, which takes an image and produces a visual result using LIBXSMM's MHD image file I/O. Performing a single convolution is *not* a showcase of LIBXSMM's Deeplearning as the code only runs over a single image with one channel.
LIBXSMM's CNNs are vectorized over image channels (multiple images) according to the native vector-width of the processor and otherwise fall back to a high-level implementation.

**Note**: For high-performance deep learning, please refer to the collection of [CNN layer samples](https://github.com/hfp/libxsmm/tree/master/samples/deeplearning/cnnlayer).

The executable can run with the following arguments (all arguments are optional):

> mhd   [&lt;filename-in&gt;  [&lt;nrepeat&gt;  [&lt;kw&gt;  [&lt;kh&gt;]  [&lt;filename-out&gt;]]]]

For stable timing (benchmark), the key operation (convolution) may be repeated (`nrepeat`). Further, `kw` and `kh` can specify the kernel-size of the convolution. The `filename-in` and `filename-out` name MHD-files used as input and output respectively. The `filename-in` may be a pseudo-file (that does not exist) but specify the image resolution of generated input (`w`[x`h`] where the file `wxh.mhd` stores the generated image data). To load an image from a familiar format (JPG, PNG, etc.), please have a look at [Meta Image File I/O](https://libxsmm.readthedocs.io/libxsmm_aux/#meta-image-file-io).

