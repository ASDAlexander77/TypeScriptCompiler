let a: string[];

class C1 {
    constructor() {
    }

    foo(x: number, y: number, ...z: string[]) {
    }
}

class D extends C1 {
    constructor() {
        super();
    }
    foo() {
        super.foo(1, 2, ...a);
    }
}

function main() {
    print("done.");
}

