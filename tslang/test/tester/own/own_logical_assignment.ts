// `a ??= b`, `a ||= b`, `a &&= b` under -mm=own: `b` is stored only where `a` does not decide, so the
// place is given `b` - a plain assignment - and not a merge of its own value and `b`. A parameter given
// a default this way (`results ??= []`, `f ??= (a) => a`) is an assigned parameter (kept), a local or a
// field takes `b` over, and the value of the expression is the place read again. Values are made at
// run time, in a loop, so a leak or a double release shows.
class Node {
    constructor(public name: string) {}
}

class Holder {
    node: Node | undefined;
    names: string[] | null = null;
    label = "";
}

function withDefault(results?: Node[]) {
    results ??= [];
    results.push(new Node("d"));
    return results.length;
}

function orDefault(results: Node[] | undefined, i: number) {
    results ||= [new Node("o" + i)];
    return results.length;
}

function callback(f?: (a: number) => number) {
    f ??= (a: number) => a;
    return f(42);
}

function andThen(n: Node | undefined, i: number) {
    n &&= new Node("and" + i);
    return n ? n.name.length : 0;
}

let made = 0;
function make(i: number) {
    made++;
    return new Node("m" + i);
}

function main() {
    const h = new Holder();
    let total = 0;
    for (let i = 0; i < 3000; i++) {
        total += withDefault() + withDefault([new Node("x" + i)]);
        total += orDefault(undefined, i) + orDefault([new Node("y" + i), new Node("z")], i);
        total += callback() + callback((a: number) => a + 1);
        total += andThen(undefined, i) + andThen(new Node("n" + i), i);

        let local: Node | undefined = undefined;
        local ??= make(i);
        local ??= make(i);
        total += local.name.length;

        h.node = undefined;
        h.node ??= make(i);
        h.node ??= make(i);
        total += h.node.name.length;

        h.names ??= ["e" + i, "f" + i];
        total += h.names.length;
        if (i % 8 == 0) {
            h.names = null;
        }

        h.label ||= "label" + i;
        const value = (h.label &&= "set" + i);
        assert(value == "set" + i && h.label == "set" + i, "the value of &&=");
    }

    assert(made == 3000 * 2, "the right side runs only where it is assigned");
    assert(total > 0, "total");
    print("done.");
}
