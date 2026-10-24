// ...a @value class as well
class Box<T> {
    constructor(public l: T | null) {}
}

@value
class L {
    constructor(public readonly v: number, public readonly box: Box<L>) {}
}

function main() {
    const bx = new Box<L>(null);
    print(new L(1, bx).v);
}
