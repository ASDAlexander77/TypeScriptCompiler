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

function* foo3() {
    let i = 1;
    while (i < 3) {
        yield ++i;
        yield i;
    }
}

function main3() {
    let count = 0;
    for (const o of foo3()) {
        assert(2 == o || 3 == o);
        count++;
    }

    assert(count == 4);
}

function* foo4() {
    for (let i = 2; i < 4; i++) {
        yield i;
        yield i;
    }
}

function main4() {
    let count = 0;
    for (const o of foo4()) {
        print(o);
        count++;
    }

    assert(count == 4);
}

function main() {
    main1();
    main2();
    main3();
    main4();
    print("done.");
}
