// -mm=own rejects: a C function of numbers (`raise`) in a module that gives a function to
// `call_intrinsic` (an identity intrinsic on a pointer, so an opaque pointer to `onSig` comes
// back), then its address to a C function of numbers (`signal(15, <u64>p)`): a function turned
// into data is C's to call, so `raise` may call it back, and it overwrites `gh.c`, which `c`
// borrows (spec 31: `inline_asm` and `call_intrinsic` given a function turn the rule off).
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
declare function signal(sig: s32, handler: u64): u64;
declare function raise(sig: s32): s32;
let gh = new H();
function onSig(sig: s32) { gh.c = new V(9); }
function use() { const c = gh.c; raise(15); churn(); return c.x; }
function main() {
    const p = call_intrinsic<Opaque>("llvm.launder.invariant.group.p0", onSig);
    signal(15, <u64>p);
    print(use()); print(gh.c.x);
}
