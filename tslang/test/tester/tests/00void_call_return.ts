// `return g()` of a call that returns nothing is the call and a `return;`: the function failed with nothing
// said ("failed statement"), also through `super` and in an arrow. Elements past the end of a tuple a literal
// is taken as are undefined (`var [a, b, c] = [1, "x"]`), not a statement failed with nothing said.
let calls = 0;

function g() {
    calls++;
}

function h() {
    return g();
}

class A {
    foo() {
        calls++;
    }
}

class B extends A {
    foo() {
        return super.foo();
    }

    bar() {
        const f = () => {
            return super.foo();
        };
        f();
    }
}

function main() {
    h();
    new B().foo();
    new B().bar();
    assert(calls == 3);

    var tuple = [1, "hello"];
    var [x, y, z] = tuple;
    assert(x == 1 && y == "hello" && z === undefined);
    print("done.");
}
