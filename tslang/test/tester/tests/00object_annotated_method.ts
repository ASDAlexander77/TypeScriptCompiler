// A type-literal annotation with a METHOD member (`inc(): void`, MethodSignature)
// must keep `this` callable through the annotated value: the member's field type
// carries an implicit opaque `this` param (same convention as interface method
// funcTypes), so property access binds the receiver. Previously the member
// degraded to a this-less funcptr: the annotation cast dropped the receiver
// ("losing this reference") and calls silently passed garbage as `this` -
// mutations went nowhere (count stayed 0).

var counterObj: { count: number; inc(): void } = { count: 0, inc() { this.count = this.count + 1; } };

function localAnnotated() {
    let c: { count: number; inc(): void } = { count: 10, inc() { this.count = this.count + 1; } };
    c.inc();
    c.inc();
    assert(c.count == 12);
    print(c.count);
}

function reader() {
    let obj: { base: number; twice(): number } = { base: 21, twice() { return this.base * 2; } };
    const v = obj.twice();
    assert(v == 42);
    print(v);
}

function main() {
    counterObj.inc();
    counterObj.inc();
    print(counterObj.count);
    assert(counterObj.count == 2);

    localAnnotated();
    reader();

    print("done.");
}
