import './export_owned_returns'

// A module can only see a declaration of an imported function, never its returns, so
// OwnedReturnConsumptionPass used to refuse to classify every one of them and their results
// were never consumed. That leaked outright - measured on a two-module program whose work is
// an imported method returning a string, `rc` held 18.0 MB against `none`'s 18.0 and `gc`'s
// 5.7, so reference counting reclaimed nothing at all across the boundary - and it also
// poisoned the candidate set for the whole member name, so a LOCAL subclass's calls stopped
// being consumed too. See docs/reference-counting-evaluation.md item 5al.
//
// This is the other side of that fix: consuming a reference the callee never handed over
// would free a live string. So the test holds on to results, allocates hard over anything
// wrongly freed, and only then reads them back - a freed block keeps its contents until
// something reuses it, which is what makes the churn load-bearing rather than decorative.

class Dog extends M.Animal {
    constructor(name: string) {
        super(name);
    }

    speak(): string {
        return `${this.name} barks.`;
    }
}

function main() {
    const a = new M.Animal("Generic");
    const d = new Dog("Mitzie");
    const asBase: M.Animal = d;
    const asIface: M.Describable = a;

    // every shape, held rather than dropped
    let kept: string[] = [];
    for (let i = 0; i < 50; i++) {
        kept.push(a.speak());          // imported method, direct
        kept.push(d.speak());          // local override - the call the imported one poisoned
        kept.push(asBase.speak());     // virtual dispatch to the local override
        kept.push(asIface.describe()); // through an imported interface
        kept.push(M.greet("world"));   // plain imported function
    }

    // reuse anything that was freed too early
    let churn = 0;
    for (let i = 0; i < 20000; i++) {
        churn = churn + a.speak().length + M.greet("x").length;
    }

    let bad = 0;
    for (let i = 0; i < 50; i++) {
        const at = i * 5;
        if (kept[at] != "Generic makes a noise.") bad = bad + 1;
        if (kept[at + 1] != "Mitzie barks.") bad = bad + 1;
        if (kept[at + 2] != "Mitzie barks.") bad = bad + 1;
        if (kept[at + 3] != "animal Generic") bad = bad + 1;
        if (kept[at + 4] != "hello world") bad = bad + 1;
    }

    assert(bad == 0, "imported call results were freed while still referenced");
    assert(churn == 20000 * ("Generic makes a noise.".length + "hello x".length), "churn");

    print("done.");
}
