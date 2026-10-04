// `length =` with a number that is not an integer stops the program (#483)
function main() {
    print("printed before");
    const a: number[] = [1, 2, 3];
    let n = 2.5;
    a.length = n;
    print("length: ", a.length);
}
