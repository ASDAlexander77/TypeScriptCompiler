// `interface X extends Object` / `extends Function`: the built-in types give the interface nothing of their own and are
// left out ("'Object' is a type here, not a value"). `export default` of an interface (a type, no value) exports
// nothing at run time ("failed to legalize operation 'ts.InterfaceRef'").
interface Server extends Object {
    name: string;
}

interface Named extends Server, Object {
    id: number;
}

interface Callback extends Function {
}

export default Server;

function main() {
    const s: Server = { name: "a" };
    const n: Named = { name: "b", id: 2 };
    assert(s.name == "a" && n.name == "b" && n.id == 2);
    print("done.");
}
