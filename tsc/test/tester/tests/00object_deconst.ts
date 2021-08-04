function main() {
    const { aa, bb } = { aa: 10, bb: 20 };
    print(aa + bb);
    assert(aa + bb == 30);

    const {
        aa2,
        bb2: { q, r },
    } = { aa2: 10, bb2: { q: 1, r: 2 } };

    assert(aa2 == 10);
    assert(q == 1);
    assert(r == 2);
    print(aa2, q, r);

    const user = {
        id: 42,
        is_verified: true,
    };

    const { id, is_verified } = user;

    print(id); // 42
    print(is_verified); // true

    assert(id == 42);
    assert(is_verified);

    // TODO: finish it
    /*
    const o = { p: 42, q: true };
    const { p: foo, q: bar } = o;

    print(foo); // 42
    print(bar); // true

    assert(foo == 42);
    assert(bar);
    */

    print("done.");
}
