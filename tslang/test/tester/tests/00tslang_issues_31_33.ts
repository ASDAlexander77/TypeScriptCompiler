// TSLANG_ISSUES 31 and 33, without threads: what Thread.spawn does to a closure and its result.
// 31: under -mm=own a `const` Shared<T> is captured by copy, so any number of closures that escape
// hold a counted handle each. 33: the result of a function called through a field is the caller's when
// every function that can be there makes its result, so it can be stored in a field.

class Box {
    constructor(public value: int) {}
}

class Keeper {
    constructor(public f: () => int) {}
}

class Start<R> {
    result: R;

    constructor(public fn: () => R) {}

    run() {
        this.result = this.fn();
    }

    runInTry() {
        try {
            const result = this.fn();
            this.result = result;
        } catch (e) {
            print("not reached");
        }
    }

    drop() {
        this.fn();
    }

    get(): R {
        return this.result;
    }
}

function spawn<R>(fn: () => R): Start<R> {
    return new Start<R>(fn);
}

function churn() {
    // same size as what may have been freed, so a use after free reads one of these
    const boxes: Box[] = [];
    for (let i = 0; i < 16; i++) {
        boxes.push(new Box(-1));
    }

    assert(boxes[15].value == -1);
}

// Shared.count is -1 under the models that do not count
function expectCount(actual: number, expected: number) {
    assert(actual == -1 || actual == expected);
}

function runBoth(a: Keeper, b: Keeper) {
    a.f();
    b.f();
}

function twoClosures(counter: Shared<Box>) {
    const handle = counter;
    const a = new Keeper((): int => { handle.value.value = handle.value.value + 1; return 0; });
    const b = new Keeper((): int => { handle.value.value = handle.value.value + 1; return 0; });
    expectCount(Shared.count(counter), 4);
    runBoth(a, b);
    churn();
    assert(handle.value.value == 2);
}

function closuresShareOne() {
    const counter = new Shared(new Box(0));
    twoClosures(counter);
    expectCount(Shared.count(counter), 1);
    assert(counter.value.value == 2);

    const a = new Keeper((): int => counter.value.value);
    const b = new Keeper((): int => counter.value.value);
    expectCount(Shared.count(counter), 3);
    assert(a.f() + b.f() == 4);
}

function closuresInLoop() {
    const counter = new Shared(new Box(5));
    const keepers: Keeper[] = [];
    for (let i = 0; i < 4; i++) {
        keepers.push(new Keeper((): int => counter.value.value));
    }

    expectCount(Shared.count(counter), 5);
    churn();
    let sum = 0;
    for (const k of keepers) {
        sum += k.f();
    }

    assert(sum == 20);
}

function closureOutlivesTheLocal(): Keeper {
    const counter = new Shared(new Box(7));
    return new Keeper((): int => counter.value.value);
}

function callClosure(f: () => int): int {
    return f();
}

function borrowingBoxes(s: Shared<Box>) {
    // a closure that does not escape: over a parameter's handle, with a `let` beside it, inside a closure
    assert(callClosure((): int => s.value.value) == 3);
    let n = 1;
    n = 2;
    assert(callClosure((): int => s.value.value + n) == 5);
    const keeper = new Keeper((): int => {
        const inner = (): int => s.value.value * 2;
        return inner();
    });
    assert(keeper.f() == 6);
}

function makeBox(): Box {
    return new Box(1);
}

function resultsKept() {
    const named = new Start<Box>(makeBox);
    named.run();
    churn();
    assert(named.get().value == 1);

    const base = 41;
    const captured = new Box(1);
    const arrow = spawn((): Box => new Box(captured.value + base));
    arrow.run();
    // the second result replaces the first
    arrow.runInTry();
    churn();
    assert(arrow.get().value == 42);
    // `captured` is the closure's from the capture on: under -mm=own a use of it here is an error
    // (own_escaping_capture_used_after_error.ts)

    // one nobody keeps is given back
    for (let i = 0; i < 1000; i++) {
        named.drop();
    }

    const text = spawn((): string => "a" + base);
    text.run();
    assert(text.get() == "a41");

    const number = spawn((): int => base + 1);
    number.run();
    assert(number.get() == 42);
}

function main() {
    closuresShareOne();
    closuresInLoop();

    const late = closureOutlivesTheLocal();
    churn();
    assert(late.f() == 7);

    const handle = new Shared(new Box(3));
    borrowingBoxes(handle);
    expectCount(Shared.count(handle), 1);

    resultsKept();

    print("done.");
}
