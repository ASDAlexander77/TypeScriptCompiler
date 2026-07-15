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

    print("done.");
}
