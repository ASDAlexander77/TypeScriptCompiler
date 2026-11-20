// A write through a union of field names needs a value every named field can hold (tsc: the intersection of
// their types): "z" is no number, so it can't be written where the key may name `a`
class C {
    a = 1;
    b = "x";
}

function pick(i: number): "a" | "b" {
    return i == 0 ? "a" : "b";
}

function main() {
    const c = new C();
    c[pick(1)] = "z";
}
