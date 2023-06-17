let glb1 = 0;
function test(this: string) {
    print(this);
    glb1++;
}

function main() {
    let s: string = "asd";
    s?.test();

    s = null;
    s?.test();

    assert(glb1 == 1);

    print("done.");
}
