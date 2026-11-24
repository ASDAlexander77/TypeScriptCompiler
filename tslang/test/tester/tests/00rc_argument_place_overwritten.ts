// -mm=rc: an argument read out of a field or an element of an object a closure can reach (a captured local's
// array, a `const` object captured by value) is borrowed by the parameter; the callee - the closure - gave that
// place a new value (`arr[0] = mk(i)`, `h.o = mk(i)`), took the element out (`pop`, `shift`, `splice`) or gave the
// captured local a new array, then read its parameter: freed memory (rc printed garbage, main too). The caller now
// holds the argument across the call. A name a destructured parameter binds (`([p]: N[]) => ...`) owns what it reads
// under rc, as a local does. A global's element given to a function that sets it was held already. gc and none
// print the same; own refuses these programs (it cannot prove the borrow outlives the write).
class N { x: number; s: string; constructor(x: number) { this.x = x; this.s = "n" + x; } }
class H { o: N; constructor(o: N) { this.o = o; } }
@noinline function mk(i: number): N { return new N(i); }
function churn() { const junk: N[] = []; for (let i = 0; i < 16; i++) junk.push(new N(100 + i)); }
function v(p: N): number { return p.x + (p.s == "n" + p.x ? 0 : 1000000); }
let G: N[] = [mk(0)];
function setG(i: number) { G[0] = mk(i); }
function gf(p: N, i: number): number { setG(i); churn(); return v(p); }
function main() {
    let t: number = 0;
    G = [mk(0)];
    let arr: N[] = [mk(0), mk(1)];
    const h = new H(mk(0));
    for (let i = 0; i < 2000; i++) {
        const fe = (p: N) => { arr[0] = mk(i); churn(); return v(p); };
        t += fe(arr[0]);
        const ff = (p: N) => { h.o = mk(i); churn(); return v(p); };
        t += ff(h.o);
        const fp = (p: N) => { arr.pop(); arr.push(mk(i)); churn(); return v(p); };
        t += fp(arr[arr.length - 1]);
        const fs = (p: N) => { arr.shift(); arr.unshift(mk(i)); churn(); return v(p); };
        t += fs(arr[0]);
        const fz = (p: N) => { arr.splice(0, 1, mk(i)); churn(); return v(p); };
        t += fz(arr[0]);
        const fr = (p: N) => { arr = [mk(i), mk(i)]; churn(); return v(p); };
        t += fr(arr[1]);
        t += gf(G[0], i);
        const fd = ([p]: N[]) => { arr[0] = mk(i); churn(); return v(p); };
        t += fd(arr);
        const fo = ({ o }: H) => { h.o = mk(i); churn(); return v(o); };
        t += fo(h);
    }
    print(t);
    assert(t == 17983005, "total");
    print("done.");
}
