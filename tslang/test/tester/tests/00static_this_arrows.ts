// `this` in an arrow made in a static member - a static method, a static field's initializer - is the class, as
// `this` around the arrow is: "can't resolve name: this" (an arrow captures `this`, and a static member has no
// variable of that name to capture).
class V {
    static n = 5;
    static viaField = () => {
        return this.n;
    };

    static viaMethod() {
        const f = () => this.n + 1;
        return f();
    }

    static nested() {
        const outer = () => {
            const inner = () => this.n * 2;
            return inner();
        };
        return outer();
    }
}

function main() {
    assert(V.viaField() == 5);
    assert(V.viaMethod() == 6);
    assert(V.nested() == 10);
    print("done.");
}
