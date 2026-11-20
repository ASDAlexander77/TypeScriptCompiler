// A set accessor whose parameter type is not the get accessor's (divergent accessor types): an assignment is
// checked against the setter's type ("type '"32"' is not assignable to type 'number'").
class T {
    private v: number = 0;
    get foo(): number { return this.v; }
    set foo(x: string | number) { this.v = typeof x == "string" ? 32 : x; }
}

function main() {
    const t = new T();
    t.foo = "32";
    assert(t.foo == 32);
    t.foo = 5;
    assert(t.foo == 5);

    print("done.");
}
