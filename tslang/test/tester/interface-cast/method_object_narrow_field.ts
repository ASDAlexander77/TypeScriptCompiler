// An object with methods cast to an interface that declares one of its fields wider (`count: 0` is an
// s32, `count: number` an f64): the cast would give the interface a clone with the field widened, and
// the methods, which take the object as it is laid out, would write the narrow value into the wide slot.
interface Counter {
    count: number;
    inc(): void;
}

function main() {
    const counterObj = { count: 0, inc() { this.count = this.count + 1; } };
    const c: Counter = counterObj;
    c.inc();
    print(c.count);
}
