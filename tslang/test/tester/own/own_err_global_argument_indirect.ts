// A call through a function value may run any code, an assignment of the global among it: the
// value read out of G cannot be lent to it.
class N {
    constructor(public x: number) {}
}

let G = new N(1);
let F: () => void = () => {};

function viaFunctionValue(n: N) {
    F();
    return n.x;
}

function main() {
    F = () => {
        G = new N(3);
    };
    G = new N(2);
    print(viaFunctionValue(G));
}
