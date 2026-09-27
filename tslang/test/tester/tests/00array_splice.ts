function main() {
    // shrink: delete more than inserted
    let a: number[] = [1, 2, 3, 4, 5];
    a.splice(1, 2);
    assert(a.length == 3, "shrink len");
    assert(a[0] == 1, "shrink 0");
    assert(a[1] == 4, "shrink 1");
    assert(a[2] == 5, "shrink 2");

    // grow: insert more than deleted
    let b: number[] = [1, 2, 3];
    b.splice(1, 1, 10, 20, 30);
    assert(b.length == 5, "grow len");
    assert(b[0] == 1, "grow 0");
    assert(b[1] == 10, "grow 1");
    assert(b[2] == 20, "grow 2");
    assert(b[3] == 30, "grow 3");
    assert(b[4] == 3, "grow 4");

    // equal: same count deleted as inserted
    let c: number[] = [1, 2, 3, 4];
    c.splice(1, 2, 99);
    assert(c.length == 3, "equal len");
    assert(c[0] == 1, "equal 0");
    assert(c[1] == 99, "equal 1");
    assert(c[2] == 4, "equal 2");

    // a negative start counts from the end; -1 used to be the largest unsigned index, and faulted
    let d: number[] = [1, 2, 3, 4];
    d.splice(-1, 1);
    assert(d.length == 3 && d[2] == 3, "negative start");

    // a start before the beginning is 0
    let e: number[] = [1, 2, 3, 4];
    e.splice(-10, 1);
    assert(e.length == 3 && e[0] == 2, "start before 0");

    // a start past the end deletes nothing
    let f: number[] = [1, 2, 3, 4];
    f.splice(10, 1);
    assert(f.length == 4, "start past end");

    // a left-out delete count removes everything from start on; it used to read past the operands
    let g: number[] = [1, 2, 3, 4];
    g.splice(2);
    assert(g.length == 2 && g[1] == 2, "no delete count");

    // a negative delete count deletes nothing
    let h: number[] = [1, 2, 3, 4];
    h.splice(1, -3);
    assert(h.length == 4, "negative delete count");

    // a `number` start, negative
    let k: number[] = [1, 2, 3, 4];
    const start: number = -2;
    k.splice(start, 1);
    assert(k.length == 3 && k[2] == 4, "number start");

    print("done.");
}
