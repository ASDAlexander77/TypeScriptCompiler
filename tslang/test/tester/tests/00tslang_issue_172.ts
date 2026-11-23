// TSLANG 172: a module `const` naming a class holds the class's value (it was "failed to legalize operation
// 'ts.ClassRef'"); a `var` used before its declaration in a function is declared before the body

class B {
    n = 1;
}

const k = B;

function one() {
    return 1;
}

function name() {
    return "n" + 1;
}

function main() {
    const b = new k();
    assert(b.n == 1);

    x = 7;
    assert(x == 7);
    var x = one();
    assert(x == 1);

    s = "early";
    assert(s == "early");
    var s = name();
    assert(s == "n1");

    print("done.");
}