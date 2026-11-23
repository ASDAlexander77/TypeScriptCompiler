// TSLANG_ISSUES 166: a module variable with a type written is that type in its own initializer, so a method there
// can read it (148: "can't resolve name: a"); an `async function*` typed AsyncGenerator<number> yields integer
// literals as numbers (it was "can't assign '() => {value:s32, ...}'").
interface Counter {
    n: number;
    inc(): number;
}

// the default library's, which the tester does not link
interface AsyncGenerator<T> {
    next(): { value: T; done: boolean };
}

const a: { v: number; m(): number } = { v: 5, m() { return a.v + 1; } };

let c: Counter = { n: 0, inc() { c.n++; return c.n; } };

async function* ag(): AsyncGenerator<number> {
    yield 1;
    yield 2;
}

let sum: number = 0;

async function run() {
    for await (const v of ag()) sum += v;
}

function main() {
    assert(a.m() == 6, "a reads itself");
    c.inc();
    assert(c.inc() == 2, "c reads itself");

    run();
    assert(sum == 3, "async generator of integer literals");
    print("done.");
}
