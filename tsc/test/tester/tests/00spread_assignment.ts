function os0() {
    let o = { x: 'hi', y: 17 }
    const o2 = { ...o, z: true };

    print(o2.x, o2.y);

    assert(o2.x == "hi");
    assert(o2.y == 17);
    assert(o2.z);
}

function const_arr0() {
    const a = [1, 2, 3];
    const b = [...a];

    print(b[0], b[1], b[2]);

    assert(b[0] == 1);
    assert(b[1] == 2);
    assert(b[2] == 3);
}

function main() {
    os0();
    const_arr0();

    print("done.");
}