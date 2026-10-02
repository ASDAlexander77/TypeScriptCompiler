// -mm=own, phase 5 rejects: a closure over a variable that escapes (returned). Its box would own
// the cell, which phase 5b does.
function makeCounter() {
    let n = 0;
    return () => ++n;
}

function main() {
    const f = makeCounter();
    print(f());
    print("done.");
}
