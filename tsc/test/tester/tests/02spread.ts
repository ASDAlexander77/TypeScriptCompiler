let a: string[] = ["asd"];

class C {
    constructor(x: number, y: number, ...z: string[]) {
        this.foo(x, y);
        this.foo(x, y, ...z);
    }
    foo(x: number, y: number, ...z: string[]) {
    }
}

class D extends C {
    constructor() {
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
}

function main() {

    let z: number[] = [1];
    let obj: X = {
	foo: function (x: number, y: number, ...z: string[]): X {
        // TODO: this -> interface
		//return this;
		return null;
	}
    };
    let xa: X[] = [obj, obj];

    foo(1, 2, "abc");
    foo(1, 2, ...a);
    foo(1, 2, ...a, "abc");

    obj.foo(1, 2, "abc");
    obj.foo(1, 2, ...a);
    obj.foo(1, 2, ...a, "abc");

    //obj.foo(1, 2, ...a).foo(1, 2, "abc");
    //obj.foo(1, 2, ...a).foo(1, 2, ...a);
    //obj.foo(1, 2, ...a).foo(1, 2, ...a, "abc");

    (obj.foo)(1, 2, "abc");
    (obj.foo)(1, 2, ...a);
    (obj.foo)(1, 2, ...a, "abc");

    //((obj.foo)(1, 2, ...a).foo)(1, 2, "abc");
    //((obj.foo)(1, 2, ...a).foo)(1, 2, ...a);
    //((obj.foo)(1, 2, ...a).foo)(1, 2, ...a, "abc");

    xa[1].foo(1, 2, "abc");
    xa[1].foo(1, 2, ...a);
    xa[1].foo(1, 2, ...a, "abc");

    (xa[1].foo)(...[1, 2, "abc"]);

    print("done.");
}