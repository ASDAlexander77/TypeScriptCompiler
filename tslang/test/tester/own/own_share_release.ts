// -mm=own, no --own-fallback: Shared.share marks every block a value reaches (the share routine is
// built under own too: own rc fallback design, section 6), and a marked block with one owner is still
// destroyed by that owner's release, once. Each shape is made, shared on this one thread, read and
// let go, many times over; what it prints is what gc prints, and nothing leaks (mm_fuzz.py leaks).
interface Named {
    name(): string;
}

class Leaf implements Named {
    constructor(public v: number, public s: string) {}
    name(): string {
        return this.s;
    }
}

class Pair {
    constructor(public left: Leaf, public right: Leaf | null) {}
}

class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

class List {
    head: Node | null = null;

    pop() {
        if (this.head) {
            this.head = this.head.next;
        }
    }
}

function shapes(i: number): number {
    let sum: number = 0;

    // a class with fields: a string, another object, an optional one
    const p = new Pair(new Leaf(i, "l" + i), new Leaf(i + 1, "r" + i));
    Shared.share(p);
    sum += p.left.v + p.right!.v + p.left.s.length;

    // an array of objects, and one of strings
    const leaves = [new Leaf(1, "a" + i), new Leaf(2, "b" + i), new Leaf(3, "c" + i)];
    Shared.share(leaves);
    leaves.pop();
    sum += leaves.length + leaves[1].v;
    const strings = ["x" + i, "y" + i];
    Shared.share(strings);
    sum += strings[0].length + strings[1].length;

    // a string
    const s = "s" + i;
    Shared.share(s);
    sum += s.length;

    // any, holding an object
    const a: any = new Leaf(4, "any" + i);
    Shared.share(a);
    sum += (<Leaf>a).v;

    // an interface over an object
    const n: Named = new Leaf(5, "named" + i);
    Shared.share(n);
    sum += n.name().length;

    // a closure over an object
    const leaf = new Leaf(6, "captured" + i);
    const f = () => leaf.v + leaf.s.length;
    Shared.share(f);
    sum += f();

    // a handle, and what it holds
    const h = new Shared(new Leaf(7, "handle" + i));
    Shared.share(h);
    sum += h.value.v;

    // a list popped after it is shared: each pop takes the next node out of the one let go
    // (ts.TakeOutOfField), whose word is its shared bit
    const l = new List();
    const first = new Node(1);
    const second = new Node(2);
    second.next = new Node(3);
    first.next = second;
    l.head = first;
    Shared.share(l);
    l.pop();
    sum += l.head!.v;
    l.pop();
    sum += l.head!.v;
    l.pop();
    sum += l.head ? 1 : 0;

    return sum;
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 1000; i++) {
        total += shapes(i);
    }

    print(total);
    print("done.");
}
