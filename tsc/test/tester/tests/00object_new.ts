let called = false;

class B {
    constructor(x: number, y: number, ...z: string[]) { print("constr", x, y); called = true; }
}

function main() {

    let dd : {
        new(x: number, y: number, ...z: string[]): B;
    } = B;

    const ss = new dd(1, 2);

    assert(called);

    print("done.");
}