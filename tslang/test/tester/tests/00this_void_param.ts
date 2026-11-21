// `this: void` says a function uses no `this`: no parameter at all - "'Void' can't be used as parameter type".
interface Foo {
    foo: string;
}

function f(this: void, x: number) {
    return x + 1;
}

function isFoo(this: void, object: { foo?: string }): object is Foo {
    return object.foo !== undefined;
}

class A {
    m(this: void, y: number) {
        return y * 2;
    }
}

function main() {
    assert(f(1) == 2);
    assert(new A().m(3) == 6);
    assert(isFoo({ foo: "x" }));
    print("done.");
}
