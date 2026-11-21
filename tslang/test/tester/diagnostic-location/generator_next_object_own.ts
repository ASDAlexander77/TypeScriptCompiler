// -mm=own: an object sent by next() is kept by the generator, whose callers (through the Generator interface)
// can't be made to give it up: an error at the yield that reads it, pointing at Shared<T>.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
}
class Box { constructor(public n: number) {} }

function* summer(): Generator<number, void, Box> {
    let total = 0;
    while (true) {
        const b = yield total;
        if (b) total += b.n;
    }
}

function main() {
    const g = summer();
    g.next(new Box(1));
}
