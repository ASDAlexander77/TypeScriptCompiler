// A @value type's field naming a generic of a type that does not exist is an error naming it (the
// unspecialized Box was taken in silence)
class Box<T> {
    constructor(public l: T | null) {}
}

@value
interface L {
    readonly v: number;
    readonly box: Box<Nope>;
}

function main() {
    print(1);
}
