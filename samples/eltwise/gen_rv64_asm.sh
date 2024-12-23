#!/bin/env bash

rm -rf *.asm
rm -rf *.elf

for i in `ls libxsmm_*.meltw`;
do
  cp $i /tmp/libxsmm_tmp.meltw
  llvm-objcopy -I binary -O elf64-littleriscv -B rv64gvc --rename-section=.data=.text,code /tmp/libxsmm_tmp.meltw /tmp/libxsmm_tmp.elf
  llvm-objdump -d --mattr=+V /tmp/libxsmm_tmp.elf > $i.asm
done

