~/llvm-project/build/build/bin/llvm-objcopy -I binary -O elf64-littleriscv -B rv64gvc --rename-section=.data=.text,code vle8_v.bin vle8_v.elf

~/llvm-project/build/build/bin/llvm-objdump -d --mattr=+V ./vle8_v.elf
