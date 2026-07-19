// Array-of-interface coverage: several DISTINCT object literals (each its own
// location-hashed storage type and its own lifted method, per
// docs/interface-vtable-simplification-design.md section 3) collected into a
// single Shape[]-typed array and dispatched through the SAME interface at a
// single call site in a loop. Exercises that each element's own vtable
// (constant, per-type) is independently correct - a bug here would typically
// show up as every element calling the FIRST element's method (a shared/
// aliased vtable) or a wrong `this` binding once mixed in a homogeneous
// array.

interface Shape {
    area(): number;
}

function main() {
    const square = {
        side: 4.0,
        area() { return this.side * this.side; },
    };

    const rectangle = {
        width: 3.0,
        height: 5.0,
        area() { return this.width * this.height; },
    };

    const circleLike = {
        radius: 2.0,
        area() { return this.radius * this.radius * 3.0; },
    };

    let shapes: Shape[] = [<Shape>square, <Shape>rectangle, <Shape>circleLike];

    let total = 0.0;
    for (let i = 0; i < shapes.length; i++) {
        total = total + shapes[i].area();
    }

    print(total);
    assert(square.area() == 16.0);
    assert(rectangle.area() == 15.0);
    assert(circleLike.area() == 12.0);
    assert(total == 43.0);

    print("done.");
}
