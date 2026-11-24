// TSLANG 179: an exported arrow const beside top-level code; a module const set from a call whose type is known only
// later; a closure in a closure of a parameter's default value; under -mm=rc a closure made by a default value; a
// non-exported name declared in two blocks of one namespace

export const twice = (x: number) => x * 2;

const a = () => b();
const b = () => 10;
const k = a();

let fromTop: number = 0;
fromTop = twice(k);

class A {
    v = 3;
    m(f = () => () => this) {
        return f()().v;
    }
    p(x = 2, f = () => () => x) {
        return f()();
    }
    q(f = () => this) {
        return f().v;
    }
}

function g(a = 5, f = () => a, s = "x" + a) {
    return f() + s.length;
}

namespace N {
    function helper() {
        return 1;
    }
    let count = 10;
    class C {
        x = 1;
    }
    export function one() {
        return helper() + count + new C().x;
    }
}

namespace N {
    function helper() {
        return 2;
    }
    let count = 20;
    class C {
        y = 2;
        helper() {
            return helper();
        }
    }
    export function two() {
        const o = { count, helper: 5 };
        return helper() + count + new C().y + new C().helper() + o.count + o.helper;
    }
}

function main() {
    assert(fromTop == 20 && twice(4) == 8 && k == 10);

    const o = new A();
    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        t += o.m() + o.p() + o.p(i) + o.q() + g() + g(i) + g(i, () => i);
    }

    // per i: 3 + 2 + i + 3 + 7 + 2 * (i + ("x" + i).length)
    assert(t == 1521280);

    assert(N.one() == 12 && N.two() == 2 + 20 + 2 + 2 + 20 + 5);

    print("done.");
}
