// A returned closure takes its capture box with it.
function counter() {
    let c: number = 0;
    return () => {
        c++;
        return c;
    };
}

function main() {
    const next = counter();
    next();
    next();
    assert(next() == 3);
    print("done.");
}
