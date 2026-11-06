// A class extending a class value known only at run time: `typeof A` may hold a subclass of A, whose layout
// is not A's, so its class can't be laid out here. A generic mixin given the class itself (`Mixin(A)`) is.
class A {
    a = 1;
}

function make(B: typeof A) {
    return class extends B {
        b = 2;
    };
}

function main() {
    const C = make(A);
    print(new C().b);
}
