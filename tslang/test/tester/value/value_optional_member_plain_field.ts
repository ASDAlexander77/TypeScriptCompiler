// @strict-null false
// An optional member of a @value type is read as the optional value a `f?: T` field stores: a
// plain `f: T` field cannot stand in for it.
@value
interface Hit {
    readonly dist: number;
}

interface J {
    f?: Hit;
}

class Req implements J {
    f: Hit = { dist: 1 };
}

function main() {
    const j: J = new Req();
    print(j.f.dist);
}
