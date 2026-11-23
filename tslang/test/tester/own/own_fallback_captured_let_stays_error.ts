// -mm=own --own-fallback=count: a parameter given an element of a local array (it borrows the element
// and cannot be returned) and then, in a loop, the value of a `const` a closure captures. Counted, the
// real run's error about that `const` (a move inside a loop of a value made outside it) was taken for
// the counted parameter's note, but the retain it was about was the captured `const`'s, which no count
// kept: it reached the lowering ("ownership inference left a retain behind" at <unknown>:0, fuzz
// 1122198). The parameter's group holds the locals declared from its reads, and a captured one cannot
// be counted, so the parameter is not counted: own's errors stay, at their places.
class V { x: number; constructor(x: number) { this.x = x; } }
class N { v: V; constructor(v: V) { this.v = v; } }
@noinline function mk(i: number): N { return new N(new V(i)); }
@noinline
function pick(p: N): N {
    const a: N[] = [mk(4), mk(7)];
    p = a[0];
    let s: number = 0;
    for (let j = 0; j < 4; j++) {
        const n = p;
        const f = (): number => n.v.x;
        s += f();
    }
    return p;
}
function main() {
    print(pick(mk(1)).v.x);
    print("done.");
}
