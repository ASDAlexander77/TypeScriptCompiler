#!/bin/sh
cd ../__build/tslang/ninja/debug
cmake --build . --config Debug -j 8
bash -f ../../../../scripts/separate_debug_info.sh ./bin/tslang
