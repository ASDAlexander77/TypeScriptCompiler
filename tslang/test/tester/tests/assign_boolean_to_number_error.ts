// a boolean given to a number is an error (TS2322): converted, `n = flag` made n 1
function main() {
    let n: number = 1;
    const flag: boolean = true;
    n = flag;
    print(n);
    print("done.");
}
