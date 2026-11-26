// TSLANG 204: an error by design (it failed with "failed statement" alone)
class D<T, U> {
    x: T;
    y: U;
}

function main() {
    const d = new D<number>();
    print("done.");
}
