// TSLANG_ISSUES 33: a result stored in a field must be the caller's; this callback gives back its
// argument.
class Box {
    constructor(public value: int) {}
}

class Apply<R> {
    result: R;

    constructor(public fn: (x: R) => R) {}

    run(x: R) {
        this.result = this.fn(x);
    }
}

function same(x: Box): Box {
    return x;
}

function main() {
    const box = new Box(1);
    const apply = new Apply<Box>(same);
    apply.run(box);
    print(apply.result.value + box.value);
}
