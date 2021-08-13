function main() {
    let v: number | undefined;
    v = 1;
    print(v);
    assert(v == 1);
    v += 2;
    print(v);
    assert(v == 3);
    v += v;
    print(v);
    assert(v == 6);
    v++;
    print(v);
    assert(v == 7);
    --v;
    print(v);
    assert(v == 6);
    print("done.");
}
