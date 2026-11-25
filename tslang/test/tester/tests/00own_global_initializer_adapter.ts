// TSLANG 190: a module `const` whose initializer gives a function value read at run time as another function
// type (the adapter's box), or values boxed into a rest parameter's array, in every memory model. Under -mm=own
// each was refused ("the initializer of global 'kept' takes a second reference"); inside a function it worked.
// The box and the rest array are temporaries of the initializer, given back after the call.
class C {
    constructor(public s: string) {}
}

function wrap<X>(x: X): { x: X } {
    return { x };
}

function call<A extends unknown[], T>(x: { x: (...args: A) => T }, ...args: A): T {
    return x.x(...args);
}

function count(...a: any[]): number {
    return a.length;
}

function firstLength(...cs: C[]): number {
    return cs[0].s.length;
}

const kept = call(wrap(<T>(x: T) => x), 1);
const added = call({ x: (n: number) => n + 1 }, 4);
const counted = count(1, "x", 2.5);
const first = firstLength(new C("abc"), new C("d"));

function main() {
    assert(<number>kept == 1, "a generic arrow through a generic");
    assert(added == 5, "a field");
    assert(counted == 3, "values boxed into a rest array");
    assert(first == 3, "new objects in a rest array");

    print("done.");
}
