// `length =` with a negative integer stops the program (#483)
function main() {
    print("printed before");
    const a: number[] = [1, 2, 3];
    let n = -1;
    a.length = n;
    print("length: ", a.length);
}
