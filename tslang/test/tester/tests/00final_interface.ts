// @strict-null false
// `@final interface`: only classes of this module implement it, so a call of one of its methods is
// one of the functions in the vtables the module made for it. With --opt_level=3 the call tests the
// pointer against all but one of them and calls the last without a test (SpeculativeDirectCallPass):
// every implementer has to be among them - a class that declares the interface, a class derived
// from one (its override through its own vtable), a class cast to it without declaring it, and
// null, whose methods throw.
@final
interface Shape {
    area(): number;
    scaled(k: number): number;
    name: string;
}

class Square implements Shape {
    name = "square";
    constructor(public side: number) {}
    area(): number { return this.side * this.side; }
    scaled(k: number): number { return this.area() * k; }
}

class Disc implements Shape {
    name = "disc";
    constructor(public r: number) {}
    area(): number { return 3 * this.r * this.r; }
    scaled(k: number): number { return this.area() * k; }
}

// derived from an implementer: its own area
class Frame extends Square {
    constructor(side: number, public hole: number) { super(side); this.name = "frame"; }
    area(): number { return this.side * this.side - this.hole * this.hole; }
}

// has the members and does not declare the interface
class Strip {
    name = "strip";
    constructor(public length: number) {}
    area(): number { return this.length; }
    scaled(k: number): number { return this.length * k; }
}

@noinline function make(kind: number): Shape {
    if (kind == 0) return new Square(2);
    if (kind == 1) return new Disc(1);
    if (kind == 2) return new Frame(3, 1);
    if (kind == 3) return new Strip(5);
    return null;
}

@noinline function total(shapes: Shape[]): number {
    let sum = 0.0;
    for (const shape of shapes) {
        sum += shape.area() + shape.scaled(10);
    }

    return sum;
}

// what a method does: -1 when it throws
@noinline function areaOf(shape: Shape): number {
    try {
        return shape.area();
    } catch (e) {
        return -1;
    }
}

function main() {
    const shapes: Shape[] = [];
    for (let kind = 0; kind < 4; kind++) {
        shapes.push(make(kind));
    }

    // 4 + 3 + 8 + 5 = 20, and ten times that
    assert(total(shapes) == 220, "sum over every implementer");

    assert(areaOf(make(0)) == 4, "a class that declares it");
    assert(areaOf(make(1)) == 3, "another");
    assert(areaOf(make(2)) == 8, "a derived class's override");
    assert(areaOf(make(3)) == 5, "a class that does not declare it");
    assert(areaOf(make(4)) == -1, "null throws");

    assert(make(2).name == "frame", "a field");

    print("done.");
}
