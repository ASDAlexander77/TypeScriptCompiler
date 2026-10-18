// A closure nested in a closure captures the cells, not the outer box: the outer box stays on
// the stack while both closures stay in the function.
function apply(f: (n: number) => void, n: number) {
    for (let i = 0; i < n; i++) {
        f(i);
    }
}

function total(n: number) {
    let t: number = 0;
    apply((i: number) => {
        apply((j: number) => {
            t += j;
        }, i);
    }, n);
    return t;
}

function main() {
    assert(total(5) == 10);
    print("done.");
}
