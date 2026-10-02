// -mm=own, phase 5 rejects: the closure borrows `a`, and is called after `a`'s block gave it back.
class C {
    constructor(public x: number) {}
}

function main() {
    let f: () => number;
    {
        let a = new C(1);
        f = () => a.x;
    }

    print(f());
}
