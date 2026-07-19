// Extends 00object_annotated_method.ts's coverage: that test only exercised
// zero-argument methods (inc(): void, twice(): number). Here the type-literal
// method members take parameters and one method calls ANOTHER method on
// `this` (chained dispatch through the same implicit-this-param mechanism
// fixed for MethodSignature tuple members).

function main() {
    let acc: { total: number; add(n: number): void; addTwice(n: number): void } = {
        total: 0,
        add(n: number) { this.total = this.total + n; },
        addTwice(n: number) { this.add(n); this.add(n); },
    };

    acc.add(3);
    assert(acc.total == 3);

    acc.addTwice(4);
    assert(acc.total == 11);
    print(acc.total);

    let calc: { base: number; scale(factor: number): number; setBase(value: number): void } = {
        base: 5,
        scale(factor: number) { return this.base * factor; },
        setBase(value: number) { this.base = value; },
    };

    const scaled = calc.scale(3);
    assert(scaled == 15);
    print(scaled);

    calc.setBase(10);
    const rescaled = calc.scale(3);
    assert(rescaled == 30);
    print(rescaled);

    print("done.");
}
