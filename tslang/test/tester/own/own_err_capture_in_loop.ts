// TSLANG_ISSUES 31, 32: an object captured by a closure that escapes, in a loop. The error names it
// and points at its use in the closure.
class Thing {
    constructor(public value: int) {}
}

class Keeper {
    constructor(public f: () => int) {}
}

function main() {
    const thing = new Thing(1);
    const keepers: Keeper[] = [];
    for (let i = 0; i < 2; i++) {
        keepers.push(new Keeper((): int => thing.value));
    }

    print(keepers[1].f());
}
