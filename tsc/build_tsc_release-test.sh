#!/bin/sh
cd ../__build/tsc/ninja/release
ctest -j18 -C Release -T test --output-on-failure -T test --output-on-failure