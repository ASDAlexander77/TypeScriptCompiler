// A value carries no type information to test.
@value
class V {
    constructor(public readonly x: number) {}
}

function main() {
    const v = new V(1);
    print(v instanceof V);
}
