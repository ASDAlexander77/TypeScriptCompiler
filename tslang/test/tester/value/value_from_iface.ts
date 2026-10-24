// A @value class is nominal: a @value interface value of the same fields is not one of its values.
@value
class V {
    constructor(public readonly x: number, public readonly y: number) {}
}

@value
interface R {
    readonly x: number;
    readonly y: number;
}

function main() {
    const r: R = { x: 1, y: 2 };
    const b: V = r;
    print(b.x);
}
