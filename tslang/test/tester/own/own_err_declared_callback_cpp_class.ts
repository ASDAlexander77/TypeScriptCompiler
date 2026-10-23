// -mm=own rejects: a C function of numbers (`raise`) in a module with a `@cpp` class whose method
// has a body (`MyL.on`): a `MyL` given to C++ (`addListener(new MyL())`) is called through its C++
// vtable, so `raise` may call `on` back, and it overwrites `gh.c`, which `c` borrows (spec 31: a
// `@cpp` class with a method of its own is not scalar). Compiled only: there is no C++
// `addListener` to link.
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
declare function raise(sig: s32): s32;
let gh = new H();
function onSig(sig: s32) { gh.c = new V(9); }
function use() { const c = gh.c; raise(15); churn(); return c.x; }
@cpp
class Listener {
    @slot(0) on(s: s32): void;
}
@cpp
class MyL extends Listener {
    on(s: s32): void { gh.c = new V(9); }
}
declare function addListener(l: Listener): void;
function main() { addListener(new MyL()); print(use()); print(gh.c.x); }
