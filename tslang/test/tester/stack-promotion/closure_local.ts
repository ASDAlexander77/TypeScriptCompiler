// A closure that never leaves the function: its capture box goes on the stack.
function apply(f: (n: number) => void, n: number) {
    for (let i = 0; i < n; i++) {
        f(i);
    }
}

function total(n: number) {
    let t: number = 0;
    apply((i: number) => {
        t += i;
    }, n);
    return t;
}

function main() {
    assert(total(10) == 45);
    print("done.");
}
