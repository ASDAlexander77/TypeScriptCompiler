# tslang issues found while working on native_ts_graphics

Issues hit while porting https://github.com/ASDAlexander77/native_ts_graphics to tslang. Every repro
should be a complete file that was run (`tslang file.ts`, JIT, Windows x64), with the tslang version.

## Open

None. Items 1-20 (collected 2026-10-12 to pre-alpha108) are fixed or answered; their write-ups are
in this file's git history. The last one, 20 (an enum `E | undefined` compared with an `E` variable
was always `false`, with "Not applicable logical operator for type: 'E'"), was a missing enum case
in the comparison lowering: regression test `tslang/test/tester/tests/00enum_compare.ts`.
