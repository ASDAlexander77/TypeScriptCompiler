// A parameter's default value is read in the closure's own body: what it names of the function around it
// is captured, as the body's reads are. It was read across the closure's isolated region ("'ts.Load' op
// using value defined outside the region").
class B {
    m() {
        return 1;
    }
}

class C extends B {
    viaSuper() {
        const g = (b = super.m()) => b;
        return g();
    }

    viaThis() {
        const g = (b = this.m() + 1) => b;
        return g();
    }
}

function local() {
    let y = 2;
    const g = (b = y) => b;
    const h = (b = y) => b + y;
    return g() + h() + h(10);
}

// a parameter whose type holds `undefined` and has a default keeps its own type: the argument given is one
// of the type without it ("'ts.Value' op failed to verify that type of 'value' matches element type of 'optional'")
function curry(foo: number | string | undefined = 2) {
    return (bar = foo) => bar;
}

function main() {
    const c = new C();
    assert(c.viaSuper() == 1);
    assert(c.viaThis() == 2);
    assert(local() == 2 + 4 + 12);
    assert(curry()() == 2);
    assert(typeof curry("a")() == "string");
    print("done.");
}
