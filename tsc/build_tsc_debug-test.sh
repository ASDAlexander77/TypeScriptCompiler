#!/bin/sh
cd ../__build/tslang/ninja/debug
ctest -j18 -C Debug -T test --output-on-failure -T test --output-on-failure