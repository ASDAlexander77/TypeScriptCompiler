import './export_class_abstract'

class Circle extends M.NamedShape {
    radius: number = 2;

    constructor() {
        super();
        this.name = "circle";
    }

    area(): number {
        return 3 * this.radius * this.radius; // fake pi=3 to keep the check integer-exact
    }
}

function main() {
    const c = new Circle();
    assert(c.area() == 12);
    assert(c.describe() == "circle:red area=12");

    // virtual dispatch of both the abstract method (area) and the concrete
    // override (describe, which itself calls the abstract method through
    // `this`) must still reach Circle's implementation through every
    // abstract-typed ancestor reference.
    const asNamedShape: M.NamedShape = c;
    assert(asNamedShape.area() == 12);
    assert(asNamedShape.describe() == "circle:red area=12");

    const asShape: M.Shape = c;
    assert(asShape.area() == 12);
    assert(asShape.describe() == "circle:red area=12");

    print("done.");
}
