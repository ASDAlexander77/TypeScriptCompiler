// get/set accessors in a type literal declare a field: of the get accessor's type, or of the set
// accessor's parameter type when there is no get accessor
type Point = {
    get x(): number;
    set x(v: number);
    get name(): string;
    set size(v: number);
};

type Box =
    | { get done(): false; value?: number }
    | { get done(): true; value: number };

function area(p: Point) {
    return p.x * p.size;
}

function main() {
    const p: Point = { x: 2, name: "p", size: 3 };
    assert(area(p) == 6);
    p.x = 5;
    assert(p.x == 5);
    assert(p.name == "p");

    const b: Box = { done: true, value: 7 };
    if (b.done) {
        assert(b.value == 7);
    }

    print("done.");
}
