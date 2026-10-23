// `let [a] = pair()` borrows the returned tuple's first field, which the end of the function releases.
class N {
    constructor(public x: number) {}
}

function mk(i: number) {
    return new N(i);
}

function pair(): [N, N] {
    return [mk(1), mk(2)];
}

function first() {
    let [a] = pair();
    return a;
}

function main() {
    print(first().x);
}

