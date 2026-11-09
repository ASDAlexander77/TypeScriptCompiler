// A method whose result type is `this` returns the object it is called on, under -mm=own (spec 78):
// one that returns another object of its type is an error, as callers would take the result for
// the object they called it on.
class A {
    n = 1;

    copy(): this {
        const c = new A();
        c.n = this.n;
        return <this>c;
    }
}

function main() {
    print(new A().copy().n);
}
