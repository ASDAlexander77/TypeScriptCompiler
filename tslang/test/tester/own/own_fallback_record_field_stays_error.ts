// -mm=own --own-fallback=count: a value put into a field of a record built by value (`{ v: n, items: [n] }`)
// after an array literal took it. rc takes the record field's count with a retain of the whole record
// when it is boxed, not a retain of `n`; own erases that retain while the record's release still goes
// through its fields. Counted, `n`'s ops were not whole and it was freed twice (fuzz 1122397). Its
// group is not counted: own's error stays, at its place.
class W { x: number; constructor(x: number) { this.x = x; } }
class V { x: number; w: W; constructor(x: number) { this.x = x; this.w = new W(x); } }
interface IY { v: V; items: V[]; }
@noinline function mv(i: number): V { return new V(i); }
@noinline function first(p: IY): number { return p.items[0].x; }
function main() {
    let s: number = 0;
    for (let i = 0; i < 300; i++) {
        const n = mv(i + 5);
        s += first({ v: n, items: [n] });
        n.w = new W(8);
    }
    print(s);
    print("done.");
}
