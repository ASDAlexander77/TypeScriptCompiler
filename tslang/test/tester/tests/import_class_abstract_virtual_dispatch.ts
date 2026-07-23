import './export_class_abstract_virtual_dispatch'

class Square extends M.Shape {
    side: number = 3;

    area(): number {
        return this.side * this.side;
    }
}

function main() {
    const s = new Square();

    // deliberately call describe() FIRST (no preceding area() call) and
    // TWICE: with the vtable slots misaligned, describe() called `.new`
    // through a double-returning signature, so the result was whatever was
    // left in XMM0 - a preceding `assert(s.area() == 9)` could leave 9.0
    // there and make a single assert pass by pure luck (this bug hid behind
    // exactly that luck in import_class_abstract.ts for a while).
    assert(s.describe() == "red area=9");
    assert(s.describe() == "red area=9");
    assert(s.area() == 9);

    const asShape: M.Shape = s;
    assert(asShape.describe() == "red area=9");
    assert(asShape.area() == 9);

    print("done.");
}
