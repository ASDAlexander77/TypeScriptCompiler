// -mm=own rejects: a C function of numbers (`raise`) in a module that declares a C global of a
// function type (`declare var hook`): a function stored into it (`hook = onSig`) is C's to call, so
// `raise` may call it back, and it overwrites `gh.c`, which `c` borrows (spec 31: the rule holds
// only in a module whose every declared symbol is scalar). Compiled only: there is no C `hook` to
// link.
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
declare function raise(sig: s32): s32;
let gh = new H();
function onSig(sig: s32) { gh.c = new V(9); }
function use() { const c = gh.c; raise(15); churn(); return c.x; }
declare var hook: (s: s32) => void;
function main() { hook = onSig; print(use()); print(gh.c.x); }
