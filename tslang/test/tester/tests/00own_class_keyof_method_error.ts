// -mm=own: a method read through a key is bound to its object, which moves it: a key set naming a method is an
// error (fields only)
class C {
    a = 1;
    m(): number {
        return 3;
    }
}

function pick(i: number): keyof C {
    return i == 0 ? "a" : "m";
}

function main() {
    const c = new C();
    const v = c[pick(0)];
    print(typeof v);
}
