// A method or getter overriding one that returns a number, with no return type of its own, returns the base's type:
// `f() { return 3; }` infers s32 (an integer literal), and its vtable slot, called as the base's `f(): number`,
// read 0 through the base.
abstract class Shape {
    abstract area(): number;
    abstract get sides(): number;
    abstract readonly corners: number;

    scale(): number {
        return 1.5;
    }
}

class Square extends Shape {
    area() { return 4; }
    get sides() { return 4; }
    get corners() { return 4; }
    override scale() { return 2; }
}

class Big extends Square {
    override area() { return 16; }
}

function main() {
    const s: Shape = new Square();
    assert(s.area() == 4 && s.sides == 4 && s.corners == 4 && s.scale() == 2, "through the base");

    const q = new Square();
    assert(q.area() == 4 && q.scale() == 2, "through the class");

    const b: Shape = new Big();
    assert(b.area() == 16 && b.sides == 4, "two levels down");

    print("done.");
}
