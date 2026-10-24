// A @value class is nominal: a literal of its fields is not one of its values (its constructor never
// ran).
@value
class V {
    constructor(public readonly x: number, public readonly y: number) {}
}

function main() {
    const a: V = { x: 1, y: 2 };
    print(a.x);
}
