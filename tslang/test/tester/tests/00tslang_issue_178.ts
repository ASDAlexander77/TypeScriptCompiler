// TSLANG 178: a module arrow const calling a later one beside top-level code (109); a destructuring `var` in a block
// read after it (118); push/unshift/splice on an array a call made (131); a class method and a function of the
// merged namespace with one name (145); a generator typed by a program's own iterator interface (153)

const a = () => b();
const b = () => 10;
let fromTop = 0;
fromTop = a();

class C {
    foo() {
        return 1;
    }
}

namespace C {
    export function foo() {
        return 2;
    }
}

interface It<T> {
    next(): { value: T; done: boolean };
}

interface Words {
    next(): { value: string; done: boolean };
}

function* g(n: number): It<number> {
    for (let i = 1; i <= n; i++) yield i;
}

function* w(): Words {
    yield "a";
    yield "b";
}

function pair(t: number[]) {
    if (t.length > 0) {
        var [x, y] = t;
    }

    return x + y;
}

function shape(o: { x: number; y: { z: string } }) {
    {
        var { x, y: { z } } = o;
    }

    return z + x;
}

function last(n: number) {
    for (let i = 0; i < n; i++) {
        var [s] = [i * 2];
    }

    return s;
}

function make(n: number) {
    const r: number[] = [];
    for (let i = 0; i < n; i++) r.push(i);
    return r;
}

function main() {
    assert(fromTop == 10 && a() == 10);

    assert(pair([1, 2]) == 3 && shape({ x: 3, y: { z: "k" } }) == "k3" && last(3) == 4);

    const o = { a: 1, b: 2 };
    assert(Object.keys(o).push("z") == 3);
    assert(make(2).push(7, 8) == 4 && make(3).unshift(1) == 4 && make(4).splice(1, 2).length == 2);

    assert(new C().foo() == 1 && C.foo() == 2);

    const it = g(3);
    let sum: number = 0;
    for (let r = it.next(); !r.done; r = it.next()) sum += r.value / 2;
    assert(sum == 3);

    const ws = w();
    assert(ws.next().value + ws.next().value == "ab");

    print("done.");
}
