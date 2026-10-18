// A closure called by name, never passed anywhere: its capture box goes on the stack.
function total(n: number) {
    let t: number = 0;
    const g = (i: number) => {
        t += i;
    };
    for (let i = 0; i < n; i++) {
        g(i);
    }
    return t;
}

function main() {
    assert(total(10) == 45);
    print("done.");
}
