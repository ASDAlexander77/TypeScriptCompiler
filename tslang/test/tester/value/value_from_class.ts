// A @value class is nominal: an object of an ordinary class with the same fields is not one of its
// values.
@value
class V {
    constructor(public readonly x: number, public readonly y: number) {}
}

class C {
    constructor(public x: number, public y: number) {}
}

function main() {
    const v: V = new C(1, 2);
    print(v.x);
}
