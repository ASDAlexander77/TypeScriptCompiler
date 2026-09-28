// Every type argument has to match, not only the first.
class Pair<A, B> {
    constructor(public a: A, public b: B) {}
}

function main() {
    const p = new Pair<number, number>(1, 2);
    const q: Pair<number, string> = p;
    print(q.b);
}
