// ...a @value class as well
@value
class N {
    constructor(public readonly v: number, public readonly next: N | null) {}
}

function main() {
    const b = new N(2, new N(1, null));
    print(b.next ? b.next.v : -1);
}
