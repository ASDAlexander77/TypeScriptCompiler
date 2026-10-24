// -mm=own rejects (spec 36): a callee that pushes the borrowed field of a record parameter into an
// array keeps what the record borrows, which its caller's `s` releases at the end of its block.
interface Thing { id(): int; }
class S implements Thing { constructor(private n: int) {} id(): int { return this.n; } }
@value
interface Hit { readonly thing: Thing; readonly d: number; }
function churn() { const keep: S[] = []; for (let i = 0; i < 1000; i++) keep.push(new S(77777)); return keep.length; }
function keep(h: Hit, g: Thing[]) { g.push(h.thing); }
function mk(t: Thing): Hit { return { thing: t, d: 1 }; }
function main() {
    const g: Thing[] = [];
    {
        const s = new S(5);
        keep(mk(s), g);
    }
    churn();
    print(g[0].id());
}
