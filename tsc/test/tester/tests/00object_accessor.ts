let obj = {
    p: 1.0,
    get value() { return this.p; },
    set value(v: number) { this.p = v; },
}

const t1 = {
    p: 'value',
    get getter() {
        return 'value';
    },
}

const t2 = {
    v: 'value',
    set setter(v: 'value') {},
}

function main() {
    assert(obj.value == 1.0);

    obj.value = 20;

    assert(obj.value == 20.0);

    print(t1.getter);
    t2.setter = 'value';
    
    print("done.");
}