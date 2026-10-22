// -mm=own rejects: a `public` method of a class that is not exported but that an exported function
// returns. Another module gets a value of the class from it and may call the method without its
// facts, so it may not return a borrow of its argument (spec 31). Unreachable from another module,
// the same method keeps its facts (own_interface_facts).
class V {
    constructor(public x: number) {}
}
class PA { public pick(v: V): V { return v; } }
export function make(): PA { return new PA(); }
function main() {
    const p = make();
    const a = new V(5);
    const b = p.pick(a);
    print(b.x, a.x);
}
