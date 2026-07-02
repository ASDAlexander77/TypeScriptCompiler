function main() {

    const arrrr = () => (m: number) => () => (n: number) => m + n;
    const e = arrrr()(3)()(4);

    print(e);

    assert(e == 7);

    print("done.");
}