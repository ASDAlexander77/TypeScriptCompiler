// An array literal of the function's own values, only read: header and elements on the stack.
function sum3(x: number, y: number, z: number) {
    const v = [x, y, z];
    let t: number = 0;
    for (let i = 0; i < v.length; i++) {
        t += v[i];
    }
    return t;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        t += sum3(i, 1, 2);
    }
    assert(t == 502500);
    print("done.");
}
