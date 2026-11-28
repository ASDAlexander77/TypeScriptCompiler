// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
class Inner { constructor(public x: number) {} }
class B { readonly inner: Inner = new Inner(1); }
class G extends B { inner: Inner = new Inner(4); swap() { this.inner = new Inner(5); } }
function f(b: B, g: G) {
    const i = b.inner;
    g.swap();
    print(i.x);
}
function main() { const g = new G(); f(g, g); }
