// An arrow with no return type that reads a get accessor (`() => this.prop`): typed once the
// accessor is declared, wherever the arrow is - a field initializer, a method or constructor
// declared before the accessor - and whether the accessor's type is given or inferred.
class A {
    n = 3;
    private _v = 0;

    fn = () => this.prop;
    typed = () => this.typedProp;
    set2 = (v: number) => {
        this.val = v;
    };

    constructor() {
        const f = () => this.prop;
        this.first = f();
    }

    first = 0;

    viaMethod() {
        const f = () => this.prop;
        return f();
    }

    paren() {
        const f = () => (this.prop);
        return f();
    }

    get prop() {
        return this.n * 2;
    }

    get typedProp(): number {
        return this.n * 3;
    }

    get val() {
        return this._v;
    }

    set val(v: number) {
        this._v = v;
    }
}

function main() {
    const a = new A();
    assert(a.fn() == 6, "a field's arrow reads a getter");
    assert(a.typed() == 9, "a getter with its type given");
    assert(a.first == 6, "an arrow in the constructor");
    assert(a.viaMethod() == 6 && a.paren() == 6, "an arrow in a method declared before the getter");

    a.n = 5;
    assert(a.fn() == 10, "reads the getter when called");

    a.set2(7);
    assert(a.val == 7, "an arrow that calls a setter");

    print("done.");
}
