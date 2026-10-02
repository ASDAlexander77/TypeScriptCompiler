// -mm=own, phase 5b rejects: `a` moves into the closure on one path only, and the frame would
// have to release it on the other.
class C {
    constructor(public x: number) {}
}

let handlers: (() => number)[] = [];

function register(keep: boolean) {
    let a = new C(1);
    if (keep) {
        handlers.push(() => a.x);
    }
}

function main() {
    register(true);
    register(false);
    print(handlers.length);
}
