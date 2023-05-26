function main() {

    const arrrr = () => (m: number) => () => (n: number) => m + n;
    const e = arrrr()(3)()(4);

    // TODO: finish it
    //print(e);

    print("done.");
}