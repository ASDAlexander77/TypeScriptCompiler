#!/bin/sh
cd ../__build/tsc-ninja
cmake --build . --config Debug -j 8
sh -f ../scripts/separate_debug_info.sh ../__build/tsc-ninja/bin/tsc
sh -f ../scripts/separate_debug_info.sh ../__build/tsc-ninja/bin/tsc-opt
sh -f ../scripts/separate_debug_info.sh ../__build/tsc-ninja/bin/tsc-translate
