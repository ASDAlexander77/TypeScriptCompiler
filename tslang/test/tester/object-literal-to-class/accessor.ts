// An object literal with an accessor assigned to a class (#494).
class C {
    x: number;
}

function main() {
    const c: C = { x: 3, get y() { return 2; } };
    print(c.x);
}
