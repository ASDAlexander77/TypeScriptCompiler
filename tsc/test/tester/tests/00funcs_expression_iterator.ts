let glb1 = 0;

function* iter(this: string) {
    for (const c of this)
        yield c;
}

function main() {

    for (const a of "asd".iter()) { glb1++; print(a); };

    assert(glb1 == 3);

    print("done.");
}
