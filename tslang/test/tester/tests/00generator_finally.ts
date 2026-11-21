// A generator's try/finally around its yields: the finally runs once, after the last turn in the try, before what
// follows. A yield in the finally itself: 00generator_yield_in_finally.ts. A
// generator declared in a finally block is a function of its own and may yield.
let log = "";

function* around() {
    try {
        yield 1;
        yield 2;
    } finally {
        log += "f";
    }
    yield 3;
}

function* declaredInFinally() {
    let inner: () => number = () => 0;
    try {
        yield 10;
    } finally {
        function* g() {
            yield 20;
            yield 30;
        }
        inner = () => {
            let s: number = 0;
            for (const v of g()) s += v;
            return s;
        };
    }
    yield inner();
}

function main() {
    let order = "";
    for (const v of around()) order += v + log + ",";
    assert(order == "1,2,3f," && log == "f", "finally once, after the try's last turn");

    let sum: number = 0;
    for (const v of declaredInFinally()) sum += v;
    assert(sum == 60, "a generator declared in a finally");

    print("done.");
}
