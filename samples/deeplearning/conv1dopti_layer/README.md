# 1D Dilated Convolutional Layer

This package contains the optimized kernels for the 1D dilated convolutional layer. 
The C++ implementation has code for both FP32 and BF16 formats.
You can run this code on AVX-512 enabled CPUs. Ex. - Cascade Lake or Cooper lake.

## Install instructions

Install PyTorch in an anaconda or virtual environment before installing the package.
Use GCC version 8.3.0 or higher.

```bash
conda activate environment              # Activate anaconda or virtual environment containing PyTorch

cd Conv1dOpti-extension/
python setup.py install                 # Install package
cd ..
```

A user can either use run.sh script to run the torch_example.py code,
or he/she can follow the following commands.

```bash
export LD_LIBRARY_PATH={LIBXSMM_ROOT/lib}           # Set LD_LIBRARY_PATH
export OMP_NUM_THREADS=28                           # Set number of threads
export KMP_AFFINITY=compact,1,0,granularity=fine    # Set KMP affinity

python torch_example.py                             # Run the pytorch example
```

In the previous example, we compare "nn.Conv1d" layer with our optimized "Conv1dOpti" layer.
The example shows how "nn.Conv1d" can be replaced with "Conv1dOpti" layer in a neural network
without requiring any other change.
The optimized python layer can be imported using "from Conv1dOpti_ext import Conv1dOpti" in python.
The example checks the accuracy of the results and calculates the computation time of both layers.

## Limitations of the current version

- Keep padding=0 in the options. The current layer doesn't do padding.
  Explicit padding is needed for the optimized convolutional layer.
  You can use the example for reference.
- Optimized convolutional layer code can only run with stride = 1.
- Similarly, apply the nonlinearity (Ex. ReLU) separately.  

To run code in BFloat16, set enable_BF16 flag to True. BFloat16 code runs only when the parameters of Input width,
number of filters and input channels to the layer are even number.
Ex. -  Filters = 16, Channels = 16, Input width = 60000, enable_BF16 = True  ------ BF16 run
If any of the previous parameter is odd number then code runs in FP32 format.

Keep batch size as multiple of ununtilized cores (Ex. - 28, 56, 84, 128 .... on a 28 core cascade lake)
for optimal performance with the Conv1dOpti layer.
Each batch will run on a seperate thread thus performance may go down if some core are not free,
or batch size is not equal to the number of free cores.
Keep the batch size as power of 2 with the MKLDNN backend (Conv1d) for optimal performance.

