// A getter alone implements an abstract field (`abstract name: T`), as TypeScript allows: a read through the base
// calls it. The field's setter slot gets a setter that throws a TypeError, as JavaScript's write to a property with
// only a getter does (a failing assert without the default library). It was an error naming a missing 'set_name'.
abstract class Shape {
    abstract name: string;
    abstract readonly sides: number;

    describe() {
        return this.name + " " + this.sides;
    }
}

class Triangle extends Shape {
    get name() { return "triangle"; }
    get sides() { return 3; }
}

class Square extends Shape {
    private n = "square";
    get name() { return this.n; }
    set name(v: string) { this.n = v; }
    get sides() { return 4; }
}

function main() {
    const t: Shape = new Triangle();
    assert(t.name == "triangle" && t.sides == 3, "read through the base");
    assert(t.describe() == "triangle 3", "read through this");

    const s: Shape = new Square();
    s.name = "box";
    assert(s.describe() == "box 4", "a getter and a setter");

    print("done.");
}
