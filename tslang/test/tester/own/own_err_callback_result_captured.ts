// TSLANG_ISSUES 33: a result stored in a field must be the caller's; this callback hands back what
// its capture owns.
class Box {
    constructor(public value: int) {}
}

class Start<R> {
    result: R;

    constructor(public fn: () => R) {}

    run() {
        this.result = this.fn();
    }
}

function main() {
    const box = new Box(1);
    const start = new Start<Box>((): Box => box);
    start.run();
    print(start.result.value + box.value);
}
