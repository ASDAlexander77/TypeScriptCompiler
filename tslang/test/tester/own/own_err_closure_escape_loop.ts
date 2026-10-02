// -mm=own, phase 5b rejects: each turn of the loop would move `a` into another closure.
class C {
    constructor(public x: number) {}
}

let handlers: (() => number)[] = [];

function register() {
    let a = new C(1);
    for (let i = 0; i < 3; i++) {
        handlers.push(() => a.x);
    }
}

function main() {
    register();
    print(handlers.length);
}
