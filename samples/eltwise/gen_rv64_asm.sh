#!/bin/env bash

file=`basename $1 .meltw`
echo $file

~/nfs_home_old/llvm-project/build/bin/llvm-objcopy -I binary -O elf64-littleriscv -B rv64gvc --rename-section=.data=.text,code $1 $file.elf

~/nfs_home_old/llvm-project/build/bin/llvm-objdump -d --mattr=+V $file.elf > $file.asm
