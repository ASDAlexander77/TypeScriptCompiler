class Inner { constructor(public x: number) {} }
class Outer { constructor(readonly inner: Inner, public n: number) {} }
class Holder { constructor(public h: Inner) {} }
function touch(hd: Holder) { hd.h = new Inner(9); }
function main() {
    const hd = new Holder(new Inner(2));
    let o = new Outer(new Inner(1), 0);
    const i = o.inner;
    const f = () => { o = new Outer(new Inner(3), 0); };
    f();
    touch(hd);
    print(i.x, o.n);
}
