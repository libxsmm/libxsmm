#!/bin/env bash

rm -rf *.asm
rm -rf *.elf

extns=('meltw' 'mxm' 'spmm')

for extn in ${extns[@]}
do
  echo $extn
  for i in `ls libxsmm_*.$extn`;
  do
    cp $i /tmp/libxsmm_tmp.$extn
    llvm-objcopy -I binary -O elf64-littleriscv -B rv64gvc --rename-section=.data=.text,code /tmp/libxsmm_tmp.$extn /tmp/libxsmm_tmp.elf
    llvm-objdump -d --mattr=+V /tmp/libxsmm_tmp.elf > $i.asm
    echo "Generated "$i".asm"
  done
done
