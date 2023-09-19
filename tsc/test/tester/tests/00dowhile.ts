function main() {
    assert(test() == 0, "failed. 1");

    print("done.");
}

function test() {
    let i = 10;
    do {
        print(i);
    } while (--i > 0);

    return i;
}
