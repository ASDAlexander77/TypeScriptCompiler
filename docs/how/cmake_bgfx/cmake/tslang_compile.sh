#!/usr/bin/env bash
# Filter C++ CPU flags that linked targets (bx/bgfx) export via PUBLIC
# compile options. tslang does not accept those flags.
set -euo pipefail

filtered=()
for arg in "$@"; do
    case "${arg}" in
        -msse4.2|-msse4.1|-mavx|-mavx2) ;;
        *) filtered+=("${arg}") ;;
    esac
done

exec "${filtered[@]}"
