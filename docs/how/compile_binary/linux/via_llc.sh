#/bin/sh
../TypeScriptCompiler/__build/tsc-ninja/bin/tsc --emit=llvm -nogc /mnt/c/temp/1.ts 2>1.ll
../TypeScriptCompiler/3rdParty/llvm-ninja/release/bin/llc --filetype=obj --relocation-model=pic -o=1.o 1.ll
gcc -o 1.out 1.o -frtti -fexceptions -lstdc++