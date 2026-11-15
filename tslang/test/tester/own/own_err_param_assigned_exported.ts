// -mm=own rejects: a function that assigns a parameter keeps it, and one another module may call
// cannot make its callers give the argument up unless its signature says so (`@keeps("x")`).
class B {
    constructor(public b: number) {}
}
export function pick(x: B) {
    if (x.b > 0) {
        x = new B(1);
    }

    return x.b;
}
function main() {
    print(pick(new B(2)));
}
