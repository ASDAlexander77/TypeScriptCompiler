// An object literal with a method passed to a class parameter (#494).
class C {
    x: number;
    f() { return 1; }
}

function take(c: C) {
    return c.x;
}

function main() {
    print(take({ x: 3, f() { return 2; } }));
}
