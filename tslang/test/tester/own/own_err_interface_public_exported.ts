// -mm=own rejects: a `public` method of an exported class reached through an interface. Another
// module can name the class and call the method without its facts, so it may not return a borrow of
// its argument; a `public` method of a class no other module can name may (own_interface_facts,
// spec 31).
class V {
    constructor(public x: number) {}
}
interface Pick { pick(v: V): V; }
export class PA implements Pick { public pick(v: V): V { return v; } }
function main() {
    const p: Pick = new PA();
    const a = new V(5);
    const b = p.pick(a);
    print(b.x, a.x);
}
