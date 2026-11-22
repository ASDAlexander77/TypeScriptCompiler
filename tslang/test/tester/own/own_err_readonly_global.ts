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
