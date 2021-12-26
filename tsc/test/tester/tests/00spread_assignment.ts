function os0() {
    let o = { x: 'hi', y: 17 }
    const o2 = { ...o, z: true };

    print(o2.x, o2.y);

    assert(o2.x == "hi");
    assert(o2.y == 17);
    assert(o2.z);
}

function main() {
    os0();

    print("done.");
}