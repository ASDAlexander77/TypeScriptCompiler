// regression coverage: generator *methods* (as opposed to top-level function* declarations)
// on object literals and classes. These share the same asteriskToken dispatch in
// mlirGenFunctionLikeDeclaration but were never exercised by the existing 00generator*.ts
// suite, which only uses top-level `function*`.

const objGen = {
    *gen() {
        yield 1;
        yield 2;
        yield 3;
    },
};

function main1() {
    let count = 0;
    let t = 1;
    for (const o of objGen.gen()) {
        assert(t++ == o);
        count++;
    }

    assert(count == 3);
}

class Counter {
    constructor(private limit: number) {}

    *gen() {
        for (let i = 0; i < this.limit; i++) {
            yield i;
        }
    }
}

function main2() {
    const c = new Counter(4);

    let count = 0;
    let t = 0;
    for (const o of c.gen()) {
        assert(t++ == o);
        count++;
    }

    assert(count == 4);
}

// manual .next() driving on a class generator method, mirroring the top-level
// function* manual-next regression (00generator_manual_next.ts).
function main3() {
    const c = new Counter(3);
    const it = c.gen();

    let r = it.next();
    assert(!r.done);
    assert(r.value == 0);

    r = it.next();
    assert(!r.done);
    assert(r.value == 1);

    r = it.next();
    assert(!r.done);
    assert(r.value == 2);

    r = it.next();
    assert(r.done);
}

// two independently-constructed instances must not share generator state.
function main4() {
    const a = new Counter(2);
    const b = new Counter(2);

    const ia = a.gen();
    const ib = b.gen();

    assert(ia.next().value == 0);
    assert(ib.next().value == 0);
    assert(ia.next().value == 1);
    assert(ib.next().value == 1);
    assert(ia.next().done);
    assert(ib.next().done);
}

function main() {
    main1();
    main2();
    main3();
    main4();

    print("done.");
}
