#!/bin/sh
cd ../__build/tslang/ninja/release
ctest -j18 -C Release -T test --output-on-failure -T test --output-on-failure