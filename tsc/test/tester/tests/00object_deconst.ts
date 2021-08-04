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

    print("done.");
}
