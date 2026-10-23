// -mm=own rejects: a C function of numbers (`raise`) in a module that calls a declared function
// through a function value (`r(box)`, `r` a local holding `creg`) with an interface that holds a
// function through its field: `Box` has only the field `cb`, whose interface `CB` has a method.
// `raise` may call it back, and it overwrites `gh.c`, which `c` borrows (spec 31). Compiled only:
// there is no C `creg` to link.
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
declare function raise(sig: s32): s32;
let gh = new H();
interface CB { call(s: s32): void; }
class CBI implements CB { call(s: s32) { gh.c = new V(9); } }
interface Box { cb: CB; }
declare function creg(b: Box): void;
function use() { const c = gh.c; raise(15); churn(); return c.x; }
function main() { const box: Box = { cb: new CBI() }; let r = creg; r(box); print(use()); print(gh.c.x); }
