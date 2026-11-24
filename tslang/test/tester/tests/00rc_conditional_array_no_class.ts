// -mm=rc: a conditional that yields a new array (`const xs: number[] = c ? [i] : []`) in a module with no function
// that returns a heap value - no class, no `new` - leaked the array every evaluation: OwnedReturnConsumptionPass
// skipped consumeConditionalResults there, and the local took a count of its own beside the branch's. A `let`, the
// assignment form, strings and nested arrays too. Every memory model (own and gc were flat already).
function main() {
    let total: number = 0;
    for (let i = 0; i < 1000; i++) {
        const xs: number[] = i % 2 == 0 ? [i] : [];
        total += xs.length;

        let ys: number[] = i % 3 == 0 ? [i, i] : [];
        total += ys.length;

        let zs: number[] = [i];
        zs = i % 2 == 1 ? [i, i, i] : [];
        total += zs.length;

        const ss: string[] = i % 2 == 0 ? ["s" + i] : [];
        total += ss.length;

        const nested: number[][] = i % 4 == 0 ? [[i]] : [];
        total += nested.length;
    }

    print(total);
    assert(total == 3418, "total");
    print("done.");
}
