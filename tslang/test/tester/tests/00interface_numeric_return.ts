// A method implementing an interface's numeric method without a return type of its own (`clean(input: number) {
// return 123; }` for `clean(input: T): T` of Field<number>) returns the interface's type, as an override returns
// its base method's: an integer literal's s32 did not match ("method signature not matching ... => s32").
interface Field<T> {
    clean(input: T): T;
}

class NumberField implements Field<number> {
    clean(input: number) {
        return 123;
    }
}

interface I {
    someMethod(): number;
}

class C implements I {
    someMethod() {
        return 0;
    }
}

function main() {
    const f: Field<number> = new NumberField();
    assert(f.clean(1) / 2 == 61.5);
    const i: I = new C();
    assert(i.someMethod() == 0);
    print("done.");
}
