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

    print("done.");
}
