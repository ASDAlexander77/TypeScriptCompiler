function main() {
    assert(test1() == 9, "failed. 1");
    assert(test2() == 10, "failed. 2");

    print("done.");
}

function test1() {
    let j = 0;
    for (let i = 0; i < 10; i++) {
        print(i);
        j = i;
    }

    return j;
}

function test2() {
    let j = 0;
    for (let i = 0; i++ < 10; ) {
        print(i);
        j = i;
    }

    return j;
}
