// -mm=own rejects: a C function of numbers (`raise`) in a module that calls a `declare class`
// method (`w.reg(onSig)`) with a function: `raise` may call it back, and it overwrites `gh.c`, which
// `c` borrows (spec 31). Compiled only: there is no C `W` to link.
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
declare function raise(sig: s32): s32;
let gh = new H();
function onSig(sig: s32) { gh.c = new V(9); }
function use() { const c = gh.c; raise(15); churn(); return c.x; }
declare class W { reg(h: (s: s32) => void): void; }
declare function makeW(): W;
function main() { const w = makeW(); w.reg(onSig); print(use()); print(gh.c.x); }
