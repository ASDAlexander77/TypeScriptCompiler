// Unboxing an `any` asks the object what it is through slot 0 of its vtable, which has to be
// `.instanceOf` for every class. A class that implements an interface kept that interface's
// vtable there instead, so `<B>a` called the interface's vtable as a function and crashed, under
// every memory model.

interface I {
    get(): number;
}

interface J {
    twice(): number;
}

class Plain {
    x: number = 1;
}

class B implements I {
    x: number = 5;
    get() {
        return this.x;
    }
}

class Both implements I, J {
    x: number = 7;
    get() {
        return this.x;
    }

    twice() {
        return this.x * 2;
    }
}

class D extends B implements J {
    y: number = 3;
    twice() {
        return this.y * 2;
    }
}

function main() {
    const p: any = new Plain();
    const b: any = new B();
    const both: any = new Both();
    const d: any = new D();

    assert((<Plain>p).x == 1);
    assert((<B>b).get() == 5);
    assert((<Both>both).twice() == 14);
    assert((<D>d).twice() == 6);
    assert((<B>d).get() == 5);

    assert(b instanceof B);
    assert(!(b instanceof D));
    assert(d instanceof B);
    assert(d instanceof D);

    print("done.");
}
