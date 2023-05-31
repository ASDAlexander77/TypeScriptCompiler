class C3 {
    //g: new (x: this) => this;
    g: new (x: this) => x;
}

function main() {

    const c3 = new C3();
    const r = c3.g(c3);

    assert(c3 == r);

    print("done.");
}
