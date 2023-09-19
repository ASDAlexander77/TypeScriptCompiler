function main() {
    let str = [1, 2, 3];
    assert(!str.every(x => x == 2), "everyfalse");
    assert(str.every(x => x > 0), "everytrue");
    print("done.");
}
