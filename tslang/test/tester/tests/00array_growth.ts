// Arrays grow by doubling; what pop and shift vacate reads as zero when length grows again.
class Box { constructor(public v: number) {} }

function fill(into: number[], n: number) { for (let i = 0; i < n; i++) into.push(i); }

function main() {
    const a: number[] = [];
    const alias = a;
    fill(a, 1000);
    assert(alias.length == 1000 && alias[0] == 0 && alias[999] == 999, "1000 pushes through an alias");

    a.pop();
    a.length = 1000;
    assert(a[999] == 0, "a popped slot reads zero when length grows again");

    const s: number[] = [1, 2, 3];
    s.shift();
    s.unshift(9);
    s.push(4);
    assert(s.length == 4 && s[0] == 9 && s[1] == 2 && s[3] == 4, "shift, unshift and push around a growth");

    const boxes: Box[] = [];
    for (let i = 0; i < 100; i++) boxes.push(new Box(i));
    for (let i = 0; i < 50; i++) boxes.pop();
    assert(boxes.length == 50 && boxes[49].v == 49, "class elements across growth and pops");

    // pop and shift on an empty array return a zeroed element and leave the length at 0
    const e: number[] = [];
    const p = e.pop();
    const sh = e.shift();
    assert(e.length == 0 && p == 0 && sh == 0, "pop and shift on an empty array");
    e.push(1);
    assert(e.length == 1 && e[0] == 1, "a push after pop and shift on an empty array");

    const u: number[] = [];
    for (let i = 0; i < 20; i++) u.unshift(i);
    assert(u.length == 20 && u[0] == 19 && u[19] == 0, "20 unshifts across growth");

    const sp: number[] = [1, 2, 3, 4, 5, 6, 7, 8];
    sp.splice(2, 4);
    assert(sp.length == 4 && sp[0] == 1 && sp[2] == 7 && sp[3] == 8, "a splice that shrinks");
    sp.length = 8;
    assert(sp[4] == 0 && sp[7] == 0, "the slots a splice vacated read zero when length grows again");
    sp.splice(1, 0, 10, 11, 12, 13, 14, 15, 16, 17, 18);
    assert(sp.length == 17 && sp[1] == 10 && sp[9] == 18 && sp[10] == 2, "a splice that inserts 9 and grows");

    // a smaller `length =` gives back the elements it drops (under rc and own a release each),
    // and what it vacated reads zero when the length grows again
    const shrink: number[] = [1, 2, 3, 4, 5, 6];
    shrink.length = 2;
    shrink.length = 6;
    assert(shrink.length == 6 && shrink[1] == 2 && shrink[2] == 0 && shrink[5] == 0, "a shrink then a regrow reads zero");

    const texts: string[] = [];
    for (let r = 0; r < 50; r++) {
        for (let i = 0; i < 10; i++) texts.push("text-" + r + "-" + i);
        texts.length = 0;
    }
    for (let i = 0; i < 10; i++) texts.push("again-" + i);
    assert(texts.length == 10 && texts[0] == "again-0" && texts[9] == "again-9", "strings dropped by length = 0");

    const kept: Box[] = [];
    for (let i = 0; i < 8; i++) kept.push(new Box(200 + i));
    kept.length = 3;
    const churn: Box[] = [];
    for (let i = 0; i < 64; i++) churn.push(new Box(900 + i));
    assert(kept.length == 3 && kept[0].v == 200 && kept[2].v == 202, "class elements kept by a shrink stay alive");
    kept.length = 8;
    assert(kept[3] == null && kept[7] == null, "class slots a shrink vacated read null");

    print("done.");
}
