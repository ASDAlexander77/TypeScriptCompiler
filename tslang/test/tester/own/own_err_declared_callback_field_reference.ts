// -mm=own rejects: a C function of numbers (`raise`) in a module that sees a class with a function
// field as a reference to integers (`<Reference<u64>>k`, then a load of the field), and gives the
// integer to a C function of numbers (`signal(15, ...)`): a function turned into data is C's to
// call, so `raise` may call it back, and it overwrites `gh.c`, which `c` borrows (spec 31: a class
// or object cast to anything but a class or an interface takes its fields along).
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
declare function signal(sig: s32, handler: u64): u64;
declare function raise(sig: s32): s32;
let gh = new H();
function onSig(sig: s32) { gh.c = new V(9); }
function use() { const c = gh.c; raise(15); churn(); return c.x; }
class K { f: (s: s32) => void = onSig; }
let k = new K();
function main() {
    const obj = <Reference<u64>>k;
    signal(15, LoadReference(obj[1]));
    print(use()); print(gh.c.x);
}
