// -mm=own rejects: a union value narrowed by `instanceof` is a view of the union's value, and storing it
// into a global moves that value, so a call through the narrowed name after the store uses a moved value.
class Item {
    constructor(public n: number) {}
}

class A {
    walk(a: Item[]): number { return a.length; }
}

class B {
    walk(a: Item[]): number { return 0; }
}

let kept: A | null = null;

function pick(i: number): A | B {
    return i < 0 ? new B() : new A();
}

function test(a: Item[], i: number) {
    const u = pick(i);
    let r: number = 0;
    if (u instanceof A) {
        kept = u;
        r += u.walk(a);
    }
    return r;
}

function main() {
    print(test([new Item(1)], 1));
}
