// -mm=own rejects: the constructor keeps its parameter, which its callers would have to give up,
// but an exported class can be built from another module, which does not.
export class Animal {
    name: string;
    constructor(name: string) {
        this.name = name;
    }
}
function main() {
    const a = new Animal("cat" + 1);
    print(a.name);
}
