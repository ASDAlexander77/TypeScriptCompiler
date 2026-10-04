// An object literal with a method returned as a class (#494).
class C {
    x: number;
    f() { return 1; }
}

function make(): C {
    return { x: 3, f() { return 2; } };
}

function main() {
    print(make().x);
}
