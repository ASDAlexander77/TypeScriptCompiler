// -mm=own rejects: `c` is never assigned, but its initializer is a call's result, a borrow of an
// element `list` owns. Marking that block immortal would make the array's free of it a silent
// leak, so `c` stays mortal, and `f` returns a borrow of it. The call goes through a function value
// `pick` returns, whose facts the global's region does not know (#638 cannot see it), so this is
// findImmortalGlobals' refusal of any call's result, not the region check. (`firstOf` says it
// lends its argument, so it is no error of its own.)
class V {
    constructor(public x: number) {}
}
const list = [new V(1)];
@lends("a")
function firstOf(a: V[]): V {
    return a[0];
}
function pick(): (a: V[]) => V {
    return firstOf;
}
const c = pick()(list);
function f() {
    return c;
}
function main() {
    print(f().x);
}
