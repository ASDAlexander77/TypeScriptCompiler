// `let [a] = pair` borrows pair's first element: giving pair a new array frees the one `a` reads.
class N {
    constructor(public x: number) {}
}

function main() {
    let pair = [new N(1), new N(2)];
    let [a] = pair;
    pair = [new N(3)];
    print(a.x);
}
