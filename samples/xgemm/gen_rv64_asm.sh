#!/bin/env bash

rm -rf *.asm
rm -rf *.elf

for i in `ls libxsmm_*`;
do
  llvm-objcopy -I binary -O elf64-littleriscv -B rv64gvc --rename-section=.data=.text,code $i $i.elf
  llvm-objdump -d --mattr=+V $i.elf > $i.asm
done

