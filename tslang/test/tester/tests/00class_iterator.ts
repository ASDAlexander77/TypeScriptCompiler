let glb1 = 0;

class C {
    v = 10;
    *iter() {
        for (let i = 0; i < 10; i++) yield this.v + i;
    }
}

function main() {
    for (const v of new C().iter()) { glb1++; print(v) };
    assert(glb1 == 10);
    print("done.");
}