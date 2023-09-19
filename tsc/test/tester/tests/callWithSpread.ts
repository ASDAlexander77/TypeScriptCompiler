//let a: string[] = ["a-0"];
let a: string[] = [];

class C {
    constructor(x: number, y: number, ...z: string[]) {
        print("C");
        this.foo(x, y);
        this.foo(x, y, ...z);
    }
    foo(x: number, y: number, ...z: string[]) {
        print("C.foo: ", "x:", x, "\ty:", y, "\tz[0]:", z.length > 0 ? z[0] : "<empty-0>", "\tz[1]:", z.length > 1 ? z[1] : "<empty-1>");
    }
}

class D extends C {
    constructor() {
        print("D");
        super(1, 2);
        super(1, 2, ...a);
    }
    foo() {
        super.foo(1, 2);
        super.foo(1, 2, ...a);
    }
}

interface X {
    //foo(x: number, y: number, ...z: string[]): X;
    foo: (x: number, y: number, ...z: string[]) => X;
}

function foo(x: number, y: number, ...z: string[]) {
    print("foo: ", "x:", x, "\ty:", y, "\tz[0]:", z.length > 0 ? z[0] : "<empty-0>", "\tz[1]:", z.length > 1 ? z[1] : "<empty-1>");
}

function main() {

    let z: number[] = [1];

    let obj0 = {
        foo(x: number, y: number, ...z: string[]) {
            print("obj0.foo: ", "x:", x, "\ty:", y, "\tz[0]:", z.length > 0 ? z[0] : "<empty-0>", "\tz[1]:", z.length > 1 ? z[1] : "<empty-1>");
        }
    };

    let obj: X = {
        foo: function (x: number, y: number, ...z: string[]): X {
            print("(obj as X).foo: ", "x:", x, "\ty:", y, "\tz[0]:", z.length > 0 ? z[0] : "<empty-0>", "\tz[1]:", z.length > 1 ? z[1] : "<empty-1>");
            return this;
        }
    };

    let xa: X[] = [obj, obj];

    foo(1, 2, "abc");
    foo(1, 2, ...a);
    foo(1, 2, ...a, "abc");

    new D().foo();

    obj0.foo(1, 2, "abc");
    obj0.foo(1, 2, ...a);
    obj0.foo(1, 2, ...a, "abc");

    obj.foo(1, 2, "abc");
    obj.foo(1, 2, ...a);
    obj.foo(1, 2, ...a, "abc");

    obj.foo(1, 2, ...a).foo(1, 2, "abc");
    obj.foo(1, 2, ...a).foo(1, 2, ...a);
    obj.foo(1, 2, ...a).foo(1, 2, ...a, "abc");

    (obj.foo)(1, 2, "abc");
    (obj.foo)(1, 2, ...a);
    (obj.foo)(1, 2, ...a, "abc");

    ((obj.foo)(1, 2, ...a).foo)(1, 2, "abc");
    ((obj.foo)(1, 2, ...a).foo)(1, 2, ...a);
    ((obj.foo)(1, 2, ...a).foo)(1, 2, ...a, "abc");

    xa[1].foo(1, 2, "abc");
    xa[1].foo(1, 2, ...a);
    xa[1].foo(1, 2, ...a, "abc");

    // TODO: finish it
    //(xa[1].foo)(...[1, 2, "abc"]);

    print("done.");
}