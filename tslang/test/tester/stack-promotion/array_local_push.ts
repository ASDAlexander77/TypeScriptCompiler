// An array that grows: its header can stay on the stack, its elements cannot.
function count(n: number) {
    const v: number[] = [n];
    for (let i = 0; i < n; i++) {
        v.push(i);
    }
    return v.length;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100; i++) {
        t += count(i);
    }
    assert(t == 5050);
    print("done.");
}
