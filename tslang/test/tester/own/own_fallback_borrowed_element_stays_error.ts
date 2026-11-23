// -mm=own --own-fallback=count: a node stored in two holders (a second owner, refused by own) that
// is also an element of an array literal which only borrows its elements (TSLANG 55): rc's retain
// for each element goes, and the array's release is shallow. Counted, the node would keep those
// retains with nothing to give them back - a leak - so it is not counted and stays an error (own
// rc fallback design, section 3: a site that cannot leave the value's rc ops whole). The same holds
// for a borrowing closure's capture, a rest argument's array and a value boxed in an `any`.
class Node { constructor(public v: number) {} }
class Holder { node: Node | null = null; }
function sumAll(...xs: Node[]) { let s: number = 0; for (const x of xs) s += x.v; return s; }
function call(f: () => number) { return f(); }

function link(h: Holder, g: Holder, n: Node) { h.node = n; g.node = n; const arr = [n, n]; return arr.length + arr[0].v; }

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new Holder(); const g = new Holder();
        sum += link(h, g, new Node(i));
        sum += h.node!.v; h.node = null; sum += g.node!.v; g.node = null;
    }
    print(sum); print("done.");
}
