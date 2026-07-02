// @noPropertyAccessFromIndexSignature: true

interface A {
    foo: string
}

interface B {
    [k: string]: string
}

interface C {
    foo: string
    //[k: string]: string
}

function main() {

    const a: A = { foo: 'str' };
    //const b: B = {};
    const c: C = { foo: 'str' };
    const d: C | undefined = c;

    // access property
    a.foo;
    a["foo"]

    // access index signature
    //b.foo;
    //b["foo"];

    // access property
    c.foo;
    c["foo"]

    // access index signature
    //c.bar;
    //c["bar"];

    // optional access property
    d?.foo;
    d?.["foo"]

    // optional access index signature
    //d?.bar;
    //d?.["bar"];

    print("done.");

}