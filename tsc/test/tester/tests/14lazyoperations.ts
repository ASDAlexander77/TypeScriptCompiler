let lazyAcc = 0;
function testLazyOps(): void {
    print("testing lazy");
    lazyAcc = 0;
    if (incrLazyAcc(10, false) && incrLazyAcc(1, true)) {
        assert(false, "");
    } else {
        assert(lazyAcc == 10, "lazy1");
    }
    assert(lazyAcc == 10, "lazy2");
    if (incrLazyAcc(100, true) && incrLazyAcc(1, false)) {
        assert(false, "");
    } else {
        assert(lazyAcc == 111, "lazy4");
    }
    lazyAcc = 0;
    if (incrLazyAcc(100, true) && incrLazyAcc(8, true)) {
        assert(lazyAcc == 108, "lazy5");
    } else {
        assert(false, "");
    }
    lazyAcc = 0;
    if (incrLazyAcc(10, true) || incrLazyAcc(1, true)) {
        assert(lazyAcc == 10, "lazy1b");
    } else {
        assert(false, "");
    }
    assert(lazyAcc == 10, "lazy2xx");
    if (incrLazyAcc(100, false) || incrLazyAcc(1, false)) {
        assert(false, "");
    } else {
        assert(lazyAcc == 111, "lazy4x");
    }
    lazyAcc = 0;
    if (incrLazyAcc(100, false) || incrLazyAcc(8, true)) {
        assert(lazyAcc == 108, "lazy5");
    } else {
        assert(false, "");
    }
    lazyAcc = 0;
    if (
        incrLazyAcc(10, true) &&
        incrLazyAcc(1, true) &&
        incrLazyAcc(100, false)
    ) {
        assert(false, "");
    } else {
        assert(lazyAcc == 111, "lazy10");
    }
    lazyAcc = 0;
    if (
        (incrLazyAcc(10, true) && incrLazyAcc(1, true)) ||
        incrLazyAcc(100, false)
    ) {
        assert(lazyAcc == 11, "lazy101");
    } else {
        assert(false, "");
    }

    lazyAcc = 0;
    assert((true ? incrLazyNum(1, 42) : incrLazyNum(10, 36)) == 42, "?:");
    assert(lazyAcc == 1, "?:0");
    assert((false ? incrLazyNum(1, 42) : incrLazyNum(10, 36)) == 36, "?:1");
    assert(lazyAcc == 11, "?:2");
    lazyAcc = 0;

    print("testing lazy done");
}

function incrLazyAcc(delta: number, res: boolean): boolean {
    lazyAcc = lazyAcc + delta;
    return res;
}

function incrLazyNum(delta: number, res: number) {
    lazyAcc = lazyAcc + delta;
    return res;
}

function main() {
    testLazyOps();
    print("done.");
}
