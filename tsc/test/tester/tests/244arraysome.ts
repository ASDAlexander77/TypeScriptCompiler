function main() {
    let str = [1.0, 2.0, 3.0];
    assert(str.some(x => x == 2), "sometrue");
    assert(!str.some(x => x < 0), "somefalse");
    print("done.");
}

