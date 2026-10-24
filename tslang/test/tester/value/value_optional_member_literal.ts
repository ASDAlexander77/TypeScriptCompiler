// @strict-null false
// An optional member of a @value type is read as the optional value a `f?: T` field stores: a
// literal whose field is a plain T (not typed by the interface) cannot stand in for it.
@value
interface Hit {
    readonly dist: number;
}

interface J {
    f?: Hit;
}

function main() {
    const h: Hit = { dist: 1 };
    const lit = { f: h };
    const j: J = lit;
    print(j.f.dist);
}
