import './export_class_values'

// a class of this module, extending one of the other
class Triple extends K.Shape {
    area() {
        return 3 * this.s * this.s;
    }
}

function areaOf(c: typeof K.Shape, s: number) {
    return new c(s).area();
}

function main() {
    assert(areaOf(K.Square, 3) == 9 && areaOf(K.Shape, 3) == 0 && areaOf(Triple, 1) == 3, "new through the value");

    const kinds: (typeof K.Shape)[] = [K.Square, Triple, K.Shape];
    let total: number = 0;
    for (const k of kinds) {
        total += k.count;
    }

    assert(total == 2 + 1 + 1, "static field through the value");

    const c: typeof K.Shape = K.Square;
    assert(c.label() == "square" && c == K.Square && c != K.Shape, "static method, equality");

    assert(K.make(K.Square, 4).area() == 16 && K.make(Triple, 2).area() == 12, "given to the other module");

    const k: typeof K.Kind = K.Kind;
    const n: typeof K.Kind = K.Concrete;
    assert(k.id() == 1 && n.id() == 2, "abstract class");

    print("done.");
}
