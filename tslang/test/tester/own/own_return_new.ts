// -mm=own, phase 0: a factory returns a fresh instance; the caller owns it and destroys it.
class Point {
    constructor(public x: number, public y: number) {}
    len2() { return this.x * this.x + this.y * this.y; }
}
function make(i: number) { return new Point(i, i); }
function main() {
    let t: number = 0;
    for (let i = 0; i < 1000000; i++) {
        const p = make(i % 10);
        t += p.len2();
    }
    assert(t == 57000000);
    print("done.");
}
