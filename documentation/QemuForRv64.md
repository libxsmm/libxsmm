# Setting up RISCV Qemu VM:

## Qemu Installation:
./configure --enable-virtfs --enable-slirp --target-list=riscv64-softmmu --prefix=[dir-name]
make
make install

Prebuild Linux Kernel and disk image can be downloaded from `https://fedoraproject.org/wiki/Architectures/RISC-V/Installing`
Local image at pcl-epyc01 is availble at `/home/raisiddh/tools/riscv-vm`

Once qemu is installed, use the following command to start the VM:

${QEMU\_INSTALLDIR}/qemu-system-riscv64 \
    -machine virt \
    -smp 4 \
    -m 4G \
    -cpu rv64,v=true,vlen=512,elen=64,vext\_spec=v1.0\
    -virtfs local,path=shared\_folder,mount\_tag=host0,security\_model=passthrough,id=host0 \
    -kernel fw\_payload-uboot-qemu-virt-smode.bin
    -nographic\
    -object rng-random,filename=/dev/urandom,id=rng0 \
    -device virtio-rng-device,rng=rng0 \
    -device virtio-blk-device,drive=hd0 \
    -drive file=Fedora-Developer-Rawhide-20210421.n.0-sda.raw,format=raw,id=hd0\
    -device virtio-net-device,netdev=usernet \
    -netdev user,id=usernet,hostfwd=tcp::10000-:22

Host ~Shared folder in the VM can be mounted using `sudo mount -t 9p -o trans=virtio host0 shared_folder -oversion=9p2000.L` command

## Emulate RVV instructino using BSC Vehave:
Download prebuild vehave binary from [1] and untar binary in $VHAVE\_INSTALL\_PATH.

Usage: `$(VEHAVE\_INSTALL\_PATH)/bin/vehave [command line]`

## Reference:

[1] https://ssh.hca.bsc.es/epi/ftp/
