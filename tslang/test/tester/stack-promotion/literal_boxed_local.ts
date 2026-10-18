// An object literal with a method is boxed into a reference; used only through its fields, it
// goes on the stack.
function sum(n: number) {
    let t: number = 0;
    for (let i = 0; i < n; i++) {
        const o = { x: i, y: i + 1, len() { return this.x + this.y; } };
        o.x = o.x * 2;
        t += o.x + o.y;
    }
    return t;
}

function main() {
    assert(sum(100) == 14950);
    print("done.");
}
