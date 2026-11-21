// `typeof this.foo` as a field's type, while the class is being declared: "Class member 'foo' can't be found" - its
// fields are not members yet. The member's own type, as its declaration gives it (annotated, or its initializer's).
class Foo {
    public foo: string = "a";
    public bar: typeof this.foo = "b";
    count = 1;
    next: typeof this.count = 2;

    m() {
        let x: typeof this.foo = "c";
        return x;
    }
}

function main() {
    const f = new Foo();
    assert(f.foo == "a" && f.bar == "b");
    assert(f.next == 2);
    assert(f.m() == "c");
    print("done.");
}
