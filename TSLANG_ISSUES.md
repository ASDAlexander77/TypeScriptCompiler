# tslang issues found while working on native_ts_graphics

Issues hit while porting https://github.com/ASDAlexander77/native_ts_graphics to tslang. Every repro
should be a complete file that was run (`tslang file.ts`, JIT, Windows x64), with the tslang version.

## Open

Items 1-23 (collected 2026-10-12 to pre-alpha113) are fixed or answered; their write-ups are in this
file's git history. The last two: 22 (`T | null` of a `@value` type whose only field is a class or
interface crashed the compiler, a regression from #760) was fixed by #763; 23 (`h && cond` with `h`
a `@cpp` class `| null` failed with "instanceof can't be used with a @cpp class") was fixed by #765,
regression test `tslang/test/tester/tests/00cpp_union_truth.ts`.
