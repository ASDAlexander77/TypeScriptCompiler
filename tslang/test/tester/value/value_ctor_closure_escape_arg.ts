// A closure that reads `this` in a @value constructor holds the slot `new` made, which dies with
// `new`'s caller: passed to a function it may be kept beyond the slot.
function call(f: () => number) {
    return f();
}

@value
class V {
    readonly a: number;
    readonly b: number;

    constructor(a: number) {
        this.a = a;
        this.b = call(() => this.a + 1);
    }
}

function main() {
    const v = new V(4);
    print(v.b);
}
