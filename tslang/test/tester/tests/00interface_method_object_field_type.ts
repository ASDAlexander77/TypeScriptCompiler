// An object with methods cast to an interface keeps one block when its fields have the interface's
// types: the methods and the interface see the same field (`count: 0.0`, `<number>0`). One whose
// narrower field a method names through `this` is an error (interface-cast/method_object_narrow_field.ts):
// a clone with the field widened was written by the methods in the narrow type, and read back as garbage.
// An object of fields only, or whose methods do not name the field (00interface_object5.ts), is still
// copied, its field widened.
interface Counter {
    count: number;
    inc(): void;
}

interface Point {
    x: number;
}

function main() {
    const a = { count: 0.0, inc() { this.count = this.count + 1; } };
    const ca: Counter = a;
    ca.inc();
    ca.inc();
    assert(ca.count == 2, "a float literal field");
    assert(a.count == 2, "the same block");

    const b = { count: <number>0, inc() { this.count = this.count + 1; } };
    const cb: Counter = b;
    cb.inc();
    assert(cb.count == 1, "a cast literal field");

    const p = { x: 3 };
    const ip: Point = p;
    assert(ip.x == 3, "an object of fields only, widened");

    print("done.");
}
