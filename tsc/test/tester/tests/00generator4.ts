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

        // TODO: if comment next line, it will cause bug in LLVM
        print(i);

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

function* foo5() {
    yield 1;

    for (const o of [2, 3]) {
        yield o;
    }

    yield 4;
}

function main5() {
    let count = 0;
    let t = 1;
    for (const o of foo5()) {
        print(o);
        count++;
        assert(t++ == o);
    }

    assert(count == 4);
}

function* foo6_2() {
    yield 2;
    yield 3;
}

function* foo6() {
    yield 1;

    for (const o of foo6_2()) {
        yield o;
    }

    yield 4;
}

function main6() {
    let count = 0;
    let t = 1;
    for (const o of foo6()) {
        print(o);
        count++;
        assert(t++ == o);
    }

    assert(count == 4);
}

function* foo7() {
    yield 1;

    yield* foo6_2();

    yield 4;
}

function main7() {
    let count = 0;
    let t = 1;
    for (const o of foo7()) {
        print(o);
        count++;
        assert(t++ == o);
    }

    assert(count == 4);
}

function* foo8(index: number) {
    while (index < 20) {
        yield index;
        index++;
    }
}

function main8() {
    let iterator = foo8(5);
    assert(iterator.next().value == 5.0);

    print(iterator.next().value == 6.0);
}

function main() {
    main1();
    main2();
    main3();
    main4();
    main5();
    main6();
    main7();
    // TODO: bug here, if you uncomment
    //main8();
    print("done.");
}
