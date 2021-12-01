# Deep Learning with GxM

## Compiling and Building GxM

1. Install Pre-requisite Libraries: Google logging module (glog), gflags, Google's data interchange format (Protobuf), OpenCV, LMDB
2. In Makefile.config, set GXM_LIBRARY_PATH variable to the path containing above libraries
3. In Makefile.config, set LIBXSMM_PATH variable to the path containing LIBXSMM library
4. Set/clear other flags in Makefile.config as required (see associated comments in Makefile.config)
5. source setup_env.sh
6. make clean; make

## Running GxM

The network topology definitions directory is "model_zoo". Currently, it contains definitions for
AlexNet (without LRN), ResNet-50, Inception v3 along with CIFAR10 and MNIST as simple test definitions.
Each topology definition is in a .prototxt file. ResNet-50 can run with "dummy data", raw JPEG image data
or with LMDB. Filenames indicate the data source along with the minibatch size. Inception v3 runs only with
compressed LMDB data.

The hyperparameter definitions for each topology are also in the corresponding directory under "model_zoo" in
a .prototxt file with the suffix "solver". For a single-node, this file is called solver.prototxt. For multi-node
the filename also contains the global minibatch size (=single node minibatch size x number of nodes);, e.g., solver_896.prototxt contains hyperparameters for MB=56 per node and 16 nodes. The "solver*" file also contains a
flag that specifies whether to start execution from a checkpoint (and thus read load weights from the "./weights"
directory) or from scratch; by default execution starts from scratch.

Optimal parallelization of Convolutional layers in LIBXSMM happens when the number of OpenMP threads = MiniBatch.
Therefore, on Xeon

```bash
export OMP_NUM_THREADS=<MiniBatch>
export KMP_AFFINITY=compact,granularity=fine,1,0
```

The command line for a training run is:

```bash
./build/bin/gxm train <topology filename> <hyperparameter filename>
```

For example:

```bash
./build/bin/gxm train model_zoo/resnet/1_resnet50_dummy_56.prototxt model_zoo/resnet/solver.prototxt
```

## Preping on RHEL 8.0 / CentOS 8.0

```bash
dnf install protobuf
wget http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/protobuf-compiler-3.5.0-7.el8.x86_64.rpm
dnf install protobuf-compiler-3.5.0-7.el8.x86_64.rpm
wget http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/protobuf-devel-3.5.0-7.el8.x86_64.rpm
dnf install protobuf-devel-3.5.0-7.el8.x86_64.rpm
dnf install lmdb
dnf install lmdb-devel
wget http://repo.okay.com.mx/centos/8/x86_64/release/opencv-devel-3.4.1-9.el8.x86_64.rpm
wget http://repo.okay.com.mx/centos/8/x86_64/release/opencv-3.4.1-9.el8.x86_64.rpm
dnf install opencv-3.4.1-9.el8.x86_64.rpm
dnf install opencv-devel-3.4.1-9.el8.x86_64.rpm
wget http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/gflags-2.1.2-6.el8.x86_64.rpm
wget http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/gflags-devel-2.1.2-6.el8.x86_64.rpm
dnf install gflags-2.1.2-6.el8.x86_64.rpm
dnf install gflags-devel-2.1.2-6.el8.x86_64.rpm
wget http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/glog-devel-0.3.5-3.el8.x86_64.rpm
wget http://mirror.centos.org/centos/8/PowerTools/x86_64/os/Packages/glog-0.3.5-3.el8.x86_64.rpm
dnf install glog-0.3.5-3.el8.x86_64.rpm
dnf install glog-devel-0.3.5-3.el8.x86_64.rpm
```

Make sure that the makefile follows the OpenCV Ver 3 path!

