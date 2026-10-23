// `let [a] = pair` borrows pair's first element, which the end of the function releases with pair.
class N {
    constructor(public x: number) {}
}

function first() {
    let pair = [new N(1), new N(2)];
    let [a] = pair;
    return a;
}

function main() {
    print(first().x);
}
