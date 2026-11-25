// TSLANG 191: `export { a, b as c }`, `export { a as default }`, `export { a } from`, `export * from`,
// `export * as ns from` and `export { default as x } from`, each module compiled on its own. They were
// "'export { ... }' and 'export ... from' statements are not supported".
import { sum, Point, seven, bump, plus, Color, M, theAnswer, twiceSum, Shape, Num } from "./export_lists_r";
import answer from "./export_lists_m";

class Square implements Shape {
    constructor(public side: Num) {}

    area(): number {
        return this.side * this.side;
    }
}

function main() {
    assert(sum(2, 3) == 5, "a renamed re-export");
    assert(plus(4, 5) == 9, "a renamed export");
    assert(new Point(1, 2).sum() == 3, "a class re-exported");
    assert(seven == 7, "a const exported under another name");
    assert(bump() == 1 && bump() == 2, "a function exported by a list");
    assert(Color.Blue == 2, "an enum exported by a list");
    assert(M.add(1, 1) == 2, "export * as ns");
    assert(answer() == 42, "export { a as default }");
    assert(theAnswer() == 42, "export { default as x } from");
    assert(twiceSum(1, 2) == 6, "the re-exporter's own export");
    assert(new Square(3).area() == 9, "an interface and a type alias exported by a list");

    print("done.");
}
