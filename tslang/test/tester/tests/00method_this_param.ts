// A written `this:` parameter on a method annotates the receiver; it is not a parameter. It was
// added next to the implicit receiver, which shifted every argument by one (#342).

class O {
    x = 1;

    m(this: O, a: number, b: number) {
        return a + b + this.x;
    }
}

interface I {
    x: number;
    m(this: I, a: number, b: number): number;
}

class C implements I {
    x: number = 10;

    m(this: I, a: number, b: number) {
        return a + b + this.x;
    }
}

// a free function keeps its `this:` as the receiver it binds (the extension-function form)
function ext(this: O, a: number) {
    return this.x + a;
}

function main() {
    const o = new O();
    assert(o.m(1, 2) == 4);

    const i: I = new C();
    assert(i.m(1, 2) == 13);

    const lit = {
        x: 100,
        m(this: { x: number }, a: number, b: number) {
            return a + b + this.x;
        }
    };
    assert(lit.m(1, 2) == 103);

    assert(o.ext(5) == 6);

    print("done.");
}
