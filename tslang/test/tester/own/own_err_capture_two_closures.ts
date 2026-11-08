// TSLANG_ISSUES 31, 32: an object captured by two closures that escape. The error names it and
// points at its use in the second closure.
class Thing {
    constructor(public value: int) {}
}

class Keeper {
    constructor(public f: () => int) {}
}

function main() {
    const thing = new Thing(1);
    const a = new Keeper((): int => thing.value);
    const b = new Keeper((): int => thing.value);
    print(a.f() + b.f());
}
