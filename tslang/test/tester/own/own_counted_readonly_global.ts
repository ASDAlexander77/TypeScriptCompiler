// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
class Inner { constructor(public x: number) {} }
class Outer { constructor(readonly inner: Inner, public n: number) {} }
class Holder { constructor(public h: Inner) {} }
function touch(hd: Holder) { hd.h = new Inner(9); }
let gl = new Outer(new Inner(1), 0);
function swap() { gl = new Outer(new Inner(7), 0); }
function main() {
    const i = gl.inner;
    swap();
    print(i.x);
}
