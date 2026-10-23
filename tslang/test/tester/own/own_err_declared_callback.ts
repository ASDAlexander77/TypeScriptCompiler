// -mm=own rejects: a C function of numbers (`raise`) whose module gave another C function a
// function of its own (`signal(15, onSig as Opaque)`): `raise` may call it back, and it overwrites
// `gh.c`, which `c` borrows. A C declaration of numbers destroys nothing only in a module none of
// whose functions escape so (spec 31).
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
declare function signal(sig: s32, handler: Opaque): Opaque;
declare function raise(sig: s32): s32;
let gh = new H();
function onSig(sig: s32) { gh.c = new V(9); }
function use() { const c = gh.c; raise(15); churn(); return c.x; }
function main() { signal(15, onSig as Opaque); print(use()); print(gh.c.x); }
