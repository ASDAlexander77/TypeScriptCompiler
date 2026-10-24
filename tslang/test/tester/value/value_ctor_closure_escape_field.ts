// A closure that reads `this` in a @value constructor holds the slot `new` made, which dies with
// `new`'s caller: stored in a field of the value it outlives the slot.
@value
class V {
    readonly a: number;
    readonly f: () => number;

    constructor(a: number) {
        this.a = a;
        this.f = () => this.a + 1;
    }
}

function main() {
    const v = new V(4);
    print(v.f());
}
