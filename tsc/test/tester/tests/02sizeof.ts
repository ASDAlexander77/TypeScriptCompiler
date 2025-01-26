function slice(start?: int, end?: int) {
    const v = start ? start : 0;
    assert(sizeof(v) != sizeof(start));

    const v2 = !start ? 0 : start;
    assert(sizeof(v2) != sizeof(start));

    return v;
}

function main() {

    slice();
    slice(10);

    print("done.");

}
