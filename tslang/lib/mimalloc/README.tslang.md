# mimalloc (vendored)

[mimalloc](https://github.com/microsoft/mimalloc) **v2.2.4** (commit `fbd8b99c2b828428947d70fdc046bb55609be93e`),
MIT licensed (see `LICENSE`). Only `include/` and `src/` are kept; the ETW manifest files under
`src/prim/windows/` and everything else of the upstream tree are left out.

Built as one translation unit (`src/static.c`) into `TypeScriptMimalloc` (see
`lib/TypeScriptMimalloc`) - the heap a Windows executable built by tslang allocates from - and into
tslang.exe, whose JIT gives it to the program it runs.

To update: replace `include/` and `src/` with the new release's, drop the ETW files again, and
update the version above.
