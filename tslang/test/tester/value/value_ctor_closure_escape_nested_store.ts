// A closure that reads `this` in a @value constructor may only be called where it was made: captured
// by another closure, it can be stored from there and outlive the slot `new` made.
let g: () => number = (): number => 0;

@value
class V {
    readonly a: number;

    constructor(a: number) {
        this.a = a;
        const f = () => this.a + 1;
        const h = () => {
            g = f;
        };
        h();
    }
}

function main() {
    const v = new V(4);
    print(v.a, g());
}
