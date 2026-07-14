function main() {
    assert(test_do() == 5, "failed. 1");
    assert(test_while() == 5, "failed. 2");
    assert(test_for() == 4, "failed. 3");
    test_for_empty();
    test_while_labeled();
    assert(test_triple_nested_unlabeled() == 394, "failed. 4");

    test_label();

    print("done.");
}

function test_do() {
    let i = 0;
    do {
        if (i == 5) break;

        if (i == 2) continue;

        print("i = ", i);

        let j = 0;
        do {
            if (j == 3) break;

            if (j == 2) continue;

            print("j = ", j);
        } while (j++ < 5);
    } while (i++ < 10);

    return i;
}

function test_while() {
    let i = 0;
    while (i++ < 10) {
        if (i == 5) break;

        if (i == 2) continue;

        print("i = ", i);

        let j = 0;
        while (j++ < 5) {
            if (j == 3) break;

            if (j == 2) continue;

            print("j = ", j);
        }
    }

    return i;
}

function test_for() {
    let j = 0;
    for (let i = 0; i < 10; i++) {
        if (i == 5) break;

        if (i == 2) continue;

        print("i = ", i);

        for (let j = 0; j < 5; j++) {
            if (j == 3) break;

            if (j == 2) continue;

            print("j = ", j);
        }

        j = i;
    }

    return j;
}

function test_for_empty() {
    for (;;) {
        break;
    }
}

function test_while_labeled() {
    let x = 0;
    let z = 0;
    labelCancelLoops: while (true) {
        print("Outer loops: " + x);
        x += 1;
        z = 1;
        while (true) {
            print("Inner loops: " + z);
            z += 1;
            if (z === 10 && x === 10) {
                print("breaking outer");
                break labelCancelLoops;
            } else if (z === 10) {
                print("breaking inner");
                break;
            }
        }

        print("breaked inner");
    }

    print("breaked outer");

    print("done.");
}

function test_triple_nested_unlabeled() {
    // an unlabeled break/continue at 3 levels of nesting must only affect its own
    // (innermost) loop, regardless of the order the lowering passes visit the loops in.
    let outerRuns = 0;
    let middleRuns = 0;
    let innerRuns = 0;
    for (let i = 0; i < 3; i++) {
        outerRuns++;
        for (let j = 0; j < 3; j++) {
            middleRuns++;
            for (let k = 0; k < 10; k++) {
                if (k == 2) continue;
                if (k == 5) break;
                innerRuns++;
            }
        }
    }

    // outer runs 3 times, middle runs 3*3=9 times, inner: each of the 9 middle
    // iterations runs k=0,1,(skip 2),3,4 then breaks at k==5 -> 4 increments each = 36
    return outerRuns * 100 + middleRuns * 10 + innerRuns / 9;
}

function test_label() {
    let c = 3;
    let a = 3;

    lbl1: {
        print("Hello1");
        if (c-- <= 0) break lbl1;
        continue lbl1;
        print("break");
        a--;
    }

    assert(c < 0);
    assert(a == 3);

    print("Hello2");
}
