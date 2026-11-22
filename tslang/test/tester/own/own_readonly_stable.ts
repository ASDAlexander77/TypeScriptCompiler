// A `readonly` field is written only by its class's constructor, so under -mm=own a borrow of it lasts
// as long as its object: a call that overwrites some other field of the same type - `touch` below,
// which own reads as "may drop any Inner at position 1" - does not end it. Off a parameter, `this`,
// a local the function owns (`let` or `const`), and down a chain of readonly fields; a field a base
// class declares readonly too. Ordinary TypeScript under every other model.
class Inner { constructor(public x: number) {} }
class Mid { constructor(readonly inner: Inner) {} }
class Outer {
    constructor(readonly inner: Inner, readonly mid: Mid, public n: number) {}
    show(hd: Holder): number { const i = this.inner; touch(hd); return i.x; }
}
class Holder { constructor(public h: Inner) {} }
class Base { readonly inner: Inner = new Inner(6); }
class Derived extends Base { n = 0; }

function touch(hd: Holder) { hd.h = new Inner(hd.h.x + 10); }

function fromParam(o: Outer, hd: Holder): number { const i = o.inner; touch(hd); return i.x; }
function fromChain(o: Outer, hd: Holder): number { const i = o.mid.inner; touch(hd); return i.x; }
function fromBase(d: Derived, hd: Holder): number { const i = d.inner; touch(hd); return i.x; }

function main() {
    const hd = new Holder(new Inner(0));
    let total = 0;
    for (let k = 0; k < 3; k++) {
        const o = new Outer(new Inner(1), new Mid(new Inner(2)), 0);
        total += fromParam(o, hd);
        total += fromChain(o, hd);
        total += o.show(hd);

        let l = new Outer(new Inner(3), new Mid(new Inner(4)), 0);
        const li = l.inner;
        touch(hd);
        total += li.x;

        const c = new Outer(new Inner(5), new Mid(new Inner(0)), 0);
        const ci = c.mid.inner;
        touch(hd);
        total += c.inner.x + ci.x;

        total += fromBase(new Derived(), hd);
    }

    assert(total == 3 * (1 + 2 + 1 + 3 + 5 + 0 + 6), "readonly borrows");
    assert(hd.h.x == 3 * 6 * 10, "touch ran");
    print("done.");
}
