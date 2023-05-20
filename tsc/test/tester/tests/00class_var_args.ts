let a: string[];

class C1 {
    constructor() {
    }

    foo(x: number, y: number, ...z: string[]) {
    }
}

class D1 extends C1 {
    constructor() {
        super();
    }
    foo() {
        super.foo(1, 2, ...a);
    }
}

class C2 {
    constructor(x: number, y: number, ...z: string[]) {
        this.foo(x, y);
        this.foo(x, y, ...z);
    }
    foo(x: number, y: number, ...z: string[]) {
    }
}

class D2 extends C2 {
    constructor() {
        super(1, 2);
        super(1, 2, ...a);
    }
    foo() {
        super.foo(1, 2);
        super.foo(1, 2, ...a);
    }
}

function main() {
    print("done.");
}

