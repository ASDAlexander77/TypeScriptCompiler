// A `let`/`const` of a function's body read by a function made before it (`const f = () => n; const n = 1;`) is one
// of the function's variables, declared before the body as a `var` is: "can't resolve name: n".
function foo() {
    const fn = () => numFilesSelected + 1;
    const numFilesSelected = 1;
    return fn();
}

function bar() {
    const a = () => b() + 1;
    const b = () => 10;
    return a();
}

function counter() {
    const next = () => {
        count++;
        return count;
    };
    let count = 0;
    next();
    return next();
}

function main() {
    assert(foo() == 2);
    assert(bar() == 11);
    assert(counter() == 2);
    print("done.");
}
