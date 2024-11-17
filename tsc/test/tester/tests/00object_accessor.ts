let obj = {
    p: 1.0,
    get value() { return this.p; },
    set value(v: number) { this.p = v; },
}

function main() {
    assert(obj.value == 1.0);

    obj.value = 20;

    assert(obj.value == 20.0);

    print("done.");
}