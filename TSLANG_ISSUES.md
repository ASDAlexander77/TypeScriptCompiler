# tslang issues found while working on native_ts_graphics

Issues hit while porting https://github.com/ASDAlexander77/native_ts_graphics to tslang. Every repro
should be a complete file that was run (`tslang file.ts`, JIT, Windows x64), with the tslang version.

## Open

Items 1-26 (collected 2026-10-12 to pre-alpha115) are fixed or answered; their write-ups are in this
file's git history. The last three, fixed by #775 (regression tests
`tslang/test/tester/tests/00tslang_issues_24_26.ts` and `ref_element_layout_error.ts`): 24 (an array
literal holding `const X = 1 << n` widened to 64-bit elements: shifts and bitwise operators now give an
`s32`, as `int` is); 25 (`cond ? [1] : []` crashed when iterated: an empty array literal now takes the
other branch's array type); 26 (`Ref(x)` of a `number` was accepted for a `Ref<f32>` parameter: a
reference to a number laid out otherwise is now an error).
