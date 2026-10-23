// -mm=own rejects: a C function of numbers (`raise`) in a module that calls a declared function
// through a function value (`sg(15, onSig)`, `sg` a local holding `signal`) with a function: `raise` may
// call it back, and it overwrites `gh.c`, which `c` borrows (spec 31).
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
declare function signal(sig: s32, handler: (s: s32) => void): Opaque;
declare function raise(sig: s32): s32;
let gh = new H();
function onSig(sig: s32) { gh.c = new V(9); }
function use() { const c = gh.c; raise(15); churn(); return c.x; }
function main() { let sg = signal; sg(15, onSig); print(use()); print(gh.c.x); }
