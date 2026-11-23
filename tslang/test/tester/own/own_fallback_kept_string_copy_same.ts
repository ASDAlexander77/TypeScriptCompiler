// -mm=own --own-fallback=count: a string element read and given to a constructor that keeps it. Own
// gives the constructor a copy (spec 22) and accepts it, so the flag must too, with no count: the
// read was counted as a value a call keeps before (fuzz 1125342).
class T { s: string; constructor(s: string) { this.s = s; } }
function make(x: number): number {
    const t = new T([x > 7 ? "k" + 3 : "k" + 5, `t${x}`][0]);
    return t.s.length;
}
function main() {
    let n: number = 0;
    for (let i = 0; i < 100; i++) { n += make(i); }
    print(n);
    print("done.");
}
