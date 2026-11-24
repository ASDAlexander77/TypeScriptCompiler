// A concise arrow (`([a = mk(i)]: N[]) => a.x`, `({ s }: N) => s.length`) with a destructured parameter returns
// without a scope exit: the names such a parameter binds (-mm=rc: they own what they read) and its default's slot
// (-mm=own) were never given back - rc +7.2 MB, own +5.4 MB per 400 runs, and the ownership verifier refused the
// body ("some path out of the function never gives back"). Given back before the return now; every model prints the
// same.
class N { x: number; s: string; constructor(x: number) { this.x = x; this.s = "n" + x; } }
@noinline function mk(i: number): N { return new N(i); }
function main() {
    let t: number = 0;
    const none: N[] = [];
    for (let i = 0; i < 300; i++) {
        const f = ([a = mk(i)]: N[]) => a.x + a.s.length;
        const g = ({ s }: N) => s.length;
        t += f(none) + g(mk(i));
    }
    print(t);
    assert(t == 47030, "total");
    print("done.");
}
