// -mm=own: a conditional expression whose one branch yields an element of a literal it holds past the
// conditional (`c ? [mk(i)][0] : o`) and whose other branch yields a local's value. The conditional has two
// results then (the element, the literal); the local is still given back at the end of its block. (Holding the
// literal first made the pass take the other branch's value for handed on by an unknown terminator, and `o` was
// never given back.)
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

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new N(1000 + i));
    }
}
function main() {
    let total: number = 0;
    let text: number = 0;
    for (let i = 0; i < 40; i++) {
        const o = mk(3);
        const p = i % 3 == 0 ? [mk(i)][0] : o;
        total += p.x + o.x;
        const s = "z" + i;
        const t = i % 2 == 0 ? ["k" + i][0] : s;
        text += t.length + s.length;
        churn();
    }

    print(total, text);
    assert(total == 471, "total");
    assert(text == 220, "text");
    print("done.");
}
