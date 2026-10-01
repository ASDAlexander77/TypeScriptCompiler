// `return` in a generator finishes it: next() gives {value: <returned>, done: true}, every later next() is
// done, and for..of stops - a `return` of a function nested in the body is that function's own
function* withValue() {
    yield "a";
    return "end";
    yield "never";
}

function* early(stop: boolean) {
    yield 1;
    if (stop) return;
    yield 2;
}

function* inLoop(limit: number) {
    for (let i = 0; ; i++) {
        if (i >= limit) return;
        yield i;
    }
}

function* nestedFunctionReturn() {
    const twice = (x: number) => { return x * 2; };
    yield twice(1);
    yield twice(2);
}

class Counter {
    count = 2;
    *values() {
        yield this.count;
        if (this.count > 1) return;
        yield 0;
    }
}

function main() {
    const it = withValue();
    let r = it.next();
    assert(r.value == "a" && !r.done, "first yield");
    r = it.next();
    assert(r.value == "end" && r.done, "return value, done");
    r = it.next();
    assert(r.done, "done after return");
    r = it.next();
    assert(r.done, "still done");

    let seen = 0;
    for (const v of withValue()) {
        seen++;
        assert(v == "a", "for..of skips the returned value");
    }
    assert(seen == 1, "for..of stops at return");

    const e = early(true);
    assert(!e.next().done, "early: yield");
    assert(e.next().done, "early: return");
    assert(e.next().done, "early: done after return");

    let sum = 0;
    for (const v of early(false)) sum += v;
    assert(sum == 3, "no return taken");

    let count = 0;
    for (const v of inLoop(3)) count++;
    assert(count == 3, "return from a loop");

    let nested = 0;
    for (const v of nestedFunctionReturn()) nested += v;
    assert(nested == 6, "nested function return");

    let fromMethod = 0;
    for (const v of new Counter().values()) fromMethod++;
    assert(fromMethod == 1, "method generator return");

    print("done.");
}
