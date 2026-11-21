// A class method the interface calls otherwise than it is declared: with a result the interface's `void` drops
// (`foo(): void` of `foo() { return "x"; }`), or without optional parameters the method has further
// (`method()` of `method(a = 4)`). The slot holds an adapter calling it as the class does - it was "method
// signature not matching".
interface IFace {
    foo(): void;
    method(): number;
}

class Camera implements IFace {
    calls = 0;

    foo() {
        this.calls++;
        return "x";
    }

    method(a: number = 4) {
        return a * 2;
    }
}

function main() {
    const c = new Camera();
    const i: IFace = c;
    i.foo();
    assert(c.calls == 1);
    assert(i.method() == 8);
    assert(c.method(5) == 10);
    print("done.");
}
