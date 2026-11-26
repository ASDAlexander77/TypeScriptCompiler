// TSLANG 204: an error by design (it failed with "failed statement" alone)
function main() {
    const m = class C<X> {
        g(x: X) {
            return x;
        }
    };
    print("done.");
}
