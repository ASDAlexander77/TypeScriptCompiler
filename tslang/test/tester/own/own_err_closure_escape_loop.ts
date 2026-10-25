// -mm=own, phase 5b rejects: each turn of the loop would move `a` into another closure.
class C {
    constructor(public x: number) {}
}

let handlers: (() => number)[] = [];

// emptied, so no keeper: what it holds may be released (a keeper's would be immortal)
function reset() {
    handlers.length = 0;
}

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
