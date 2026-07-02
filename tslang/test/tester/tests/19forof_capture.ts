let glb1 = 0;

type func = () => void;

function testForOf() {
    let f = [];
    glb1 = 0;
    for (const q of [1, 12]) {
        f.push(<any>(() => {
            glb1 += q;
        }));
    }

    print("calling...");

    (<func>f[0])();
    (<func>f[1])();
    assert(glb1 == 13, "foc");
}

function main() {
    testForOf();
    print("done.");
}
