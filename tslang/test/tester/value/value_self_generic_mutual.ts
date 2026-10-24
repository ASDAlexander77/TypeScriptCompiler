// ...or through another @value type it names in a type argument, declared after it, with the generic
// declared first (the unspecialized Box was taken and the compiler crashed at the literal)
class Box<T> {
    constructor(public l: T | null) {}
}

@value
interface A {
    readonly v: number;
    readonly p: Box<B>;
}

@value
interface B {
    readonly w: number;
    readonly a: A;
}

function main() {
    const bx = new Box<B>(null);
    const a: A = { v: 1, p: bx };
    print(a.v);
}
