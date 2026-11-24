// -mm=rc: an argument read out of a field or an element of an object another name may reach, given to a call that
// may write that place through the other name, is held by the caller across the call. The reviewer's X01-X21: the
// object a closure captures (by reference or by value), also stored in a global (`G = h; gw2(h.o)` with `gw2` setting
// `G.o`), in another argument's container (`aw(bx, arr[0])`, `bx = [arr]`), behind a second const a closure pops
// (`const b2 = arr`), reached through a captured array (`arrh = [h]`, a closure setting `arrh[0].o`), a parameter's
// object, a callback. rc read freed memory (main too); each part now prints what gc prints. The hold needs both: an
// object on the way that may be aliased, and a call that may write a place of that kind (a field's name, an element,
// a captured local's cell) - writes that only make an object (a constructor's) do not count. gc and none print the
// same; own refuses these programs.
class N {
    x: number;
    s: string;
    constructor(x: number) {
        this.x = x;
        this.s = "n" + x;
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

@noinline function ms(i: number): string {
    return "s" + i;
}

@noinline function take(o: N | null): number {
    return o ? o.x + o.s.length : 0;
}

@noinline function takes(s: string | null): number {
    return s ? s.length : 0;
}

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 16; i++) {
        junk.push(new N(i));
    }
}
class H { o: N = mk(0); arr: N[] = [mk(0)]; h2: H | null = null; m(p: N): number { this.o = mk(77); this.arr[0] = mk(78); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; } }
let G: H = new H();
const GA: N[] = [mk(0)];
function gw(p: N): number { G.o = mk(5); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }
function gw2(p: N): number { G.o = mk(5); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }
function hw(b: H, p: N): number { b.h2!.o = mk(5); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }
function aw(bx: N[][], p: N): number { bx[0][0] = mk(5); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }
function gaw(p: N): number { GA[0] = mk(5); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }
function gaw2(p: N): number { GA.pop(); GA.push(mk(5)); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }
function cw(cb: () => void, p: N): number { cb(); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }
function pw(h: H): number { return inner(h, h.o); } function inner(h: H, p: N): number { h.o = mk(9); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }
function x01(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const arr: N[] = [mk(i)]; const f = (p: N) => { arr[0] = mk(i + 1); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(arr[0]);
    }
    return total;
}
function x02(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const h = new H(); h.o = mk(i); const f = (p: N) => { h.o = mk(i + 1); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(h.o);
    }
    return total;
}
function x03(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const h = new H(); h.o = mk(i); total += h.m(h.o);
    }
    return total;
}
function x04(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const h = new H(); h.arr[0] = mk(i); total += h.m(h.arr[0]);
    }
    return total;
}
function x05(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        G.o = mk(i); total += gw(G.o);
    }
    return total;
}
function x06(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const h = new H(); h.o = mk(i); G = h; total += gw2(h.o);
    }
    return total;
}
function x07(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const h = new H(); h.o = mk(i); const box = new H(); box.h2 = h; total += hw(box, h.o);
    }
    return total;
}
function x08(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const arr: N[] = [mk(i)]; const bx: N[][] = [arr]; total += aw(bx, arr[0]);
    }
    return total;
}
function x09(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const arr: N[] = [mk(i)]; const b2 = arr; const f = (p: N) => { b2.pop(); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(arr[0]);
    }
    return total;
}
function x10(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const arr: N[] = [mk(i)]; const f = (p: N) => { arr.shift(); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(arr[0]);
    }
    return total;
}
function x11(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const arr: N[] = [mk(i), mk(1)]; const f = (p: N) => { arr.splice(0, 1); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(arr[0]);
    }
    return total;
}
function x12(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        let arr: N[] = [mk(i)]; const f = (p: N) => { arr = [mk(3)]; let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(arr[0]);
    }
    return total;
}
function x13(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        GA[0] = mk(i); total += gaw(GA[0]);
    }
    return total;
}
function x14(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        GA[0] = mk(i); total += gaw2(GA[0]);
    }
    return total;
}
function x15(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const h = new H(); h.o = mk(i); const w = () => { h.o = mk(i + 2); }; const f = (p: N) => { w(); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(h.o);
    }
    return total;
}
function x16(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const arr: N[] = [mk(i)]; total += cw(() => { arr[0] = mk(4); }, arr[0]);
    }
    return total;
}
function x17(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const h = new H(); h.o = mk(i); total += pw(h);
    }
    return total;
}
function x18(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const h = new H(); h.o = mk(i); const arrh: H[] = [h]; const f = (p: N) => { arrh[0].o = mk(3); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(h.o);
    }
    return total;
}
function x19(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const h = new H(); h.o = mk(i); const arrh: H[] = [h]; const f = (p: N) => { arrh[0].o = mk(3); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(arrh[0].o);
    }
    return total;
}
function x20(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const arr: N[] = [mk(i)]; const f = ([p]: N[]) => { arr[0] = mk(i + 1); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(arr);
    }
    return total;
}
function x21(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const h = new H(); h.o = mk(i); const f = ({ o: p }: H) => { h.o = mk(i + 1); let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }; total += f(h);
    }
    return total;
}
// an array put into another, given with it to a callee that shortens it (`a.length = 0` drops the element)
function f4(p: N, a: N[]): number { a.length = 0; let t: number = 0; churn(); t += p.x + (p.s == "n" + p.x ? 0 : 1000000); return t; }
function z04(): number {
    let total: number = 0;
    for (let i = 0; i < 300; i++) {
        const arr: N[] = [mk(i)]; const al = [arr]; total += f4(arr[0], al[0]);
    }
    return total;
}
function main() {
    G = new H();
    GA.length = 0;
    GA.push(mk(0));
    const totals: number[] = [];
    totals.push(x01());
    totals.push(x02());
    totals.push(x03());
    totals.push(x04());
    totals.push(x05());
    totals.push(x06());
    totals.push(x07());
    totals.push(x08());
    totals.push(x09());
    totals.push(x10());
    totals.push(x11());
    totals.push(x12());
    totals.push(x13());
    totals.push(x14());
    totals.push(x15());
    totals.push(x16());
    totals.push(x17());
    totals.push(x18());
    totals.push(x19());
    totals.push(x20());
    totals.push(x21());
    totals.push(z04());
    let sum: number = 0;
    for (const t of totals) {
        sum += t;
    }

    print(sum);
    assert(sum == 986700, "total");
    print("done.");
}
