let c = 0;

function* foo1() {
    c++;
    yield 1;
    c++;
    yield 2;
    c++;
}

function main1() {
    for (const o of foo1()) {
        print(o);
    }

    assert(c == 3);
}

function* foo2() {
    let i = 1;
    yield ++i;
    yield ++i;
}

function main2() {
    let t = 2;
    let count = 0;
    for (const o of foo2()) {
        assert(t++ == o);
        count++;
    }

    assert(count == 2);
}

function main() {
    main1();
    main2();
    print("done.");
}
