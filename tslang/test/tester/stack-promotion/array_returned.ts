// A returned array outlives the function.
function pair(a: number, b: number) {
    const v = [a, b];
    return v;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100; i++) {
        const p = pair(i, 1);
        t += p[0] + p[1];
    }
    assert(t == 5050);
    print("done.");
}
