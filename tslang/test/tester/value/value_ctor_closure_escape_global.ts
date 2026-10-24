// A closure that reads `this` in a @value constructor holds the slot `new` made, which dies with
// `new`'s caller: stored in a global it outlives the slot.
let g: () => number = (): number => 0;

@value
class V {
    readonly a: number;

    constructor(a: number) {
        this.a = a;
        g = () => this.a + 1;
    }
}

function main() {
    const v = new V(4);
    print(v.a, g());
}
