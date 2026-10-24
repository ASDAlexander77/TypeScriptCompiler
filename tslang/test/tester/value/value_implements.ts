// A value has no vtable, so it cannot be seen through an interface.
interface I {
    m(): number;
}

@value
class V implements I {
    constructor(public readonly x: number) {}
    m() {
        return this.x;
    }
}

function main() {
}
