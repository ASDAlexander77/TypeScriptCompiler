#!/bin/sh
cd ../__build/tsc-ninja
cmake --build . --config Debug -j 8
bash -f ../scripts/separate_debug_info.sh ../__build/tsc-ninja/bin/tsc
bash -f ../scripts/separate_debug_info.sh ../__build/tsc-ninja/bin/tsc-opt
bash -f ../scripts/separate_debug_info.sh ../__build/tsc-ninja/bin/tsc-translate
