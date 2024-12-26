// @strict-null false
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

    // ref to method
    const m = "asd-m"?.test;
    m();

    let m2: () => void;
    m2 = m;
    m2();

    const test = (names: string[]) => names?.filter(x => x);
    for (const s of test(["asd", "asd1", null])) print(s);

    print("done.");
}
