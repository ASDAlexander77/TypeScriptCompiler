// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
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
