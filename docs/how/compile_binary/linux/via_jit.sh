#/bin/sh
../TypeScriptCompiler/__build/tsc-ninja/bin/tsc --emit=jit -nogc -dump-object-file -object-filename=1.o /mnt/c/temp/1.ts
gcc -o 1.out 1.o -frtti -fexceptions -lstdc++