import './export_class_implements_interface_abstract'

class Square extends M.Shape {
    side: number = 3;

    area(): number {
        return this.side * this.side;
    }
}

function main() {
    const s = new Square();
    assert(s.area() == 9);
    assert(s.describe() == "red area=9");

    const asDescribable: M.Describable = s;
    assert(asDescribable.describe() == "red area=9");

    const asShape: M.Shape = s;
    assert(asShape.describe() == "red area=9");
    assert(asShape.area() == 9);

    print("done.");
}
