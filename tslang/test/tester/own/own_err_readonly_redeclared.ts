class Inner { constructor(public x: number) {} }
class B { readonly inner: Inner = new Inner(1); }
class G extends B { inner: Inner = new Inner(4); swap() { this.inner = new Inner(5); } }
function f(b: B, g: G) {
    const i = b.inner;
    g.swap();
    print(i.x);
}
function main() { const g = new G(); f(g, g); }
