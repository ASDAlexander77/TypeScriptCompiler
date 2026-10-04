// An object literal with a method assigned to a class (#494): the cast reinterpreted the pointer,
// so c.x read garbage and c.f() crashed.
class C {
    x: number;
    f() { return 1; }
}

function main() {
    const c: C = { x: 3, f() { return 2; } };
    print(c.x, c.f());
}
