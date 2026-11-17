// own spec 93 across a module boundary: the library side. See import_weak_field.ts. @final: another
// module may call depth() on a borrowed Node, which an override elsewhere could make free things.
@final
export class Node {
    @weak parent: Node | null = null;
    children: Node[] = [];
    constructor(public v: number) {}

    @keeps("c")
    add(c: Node) {
        c.parent = this;
        this.children.push(c);
    }

    depth(): number {
        return this.parent ? 1 + this.parent.depth() : 0;
    }
}
