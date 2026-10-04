// `length =` past 2^32 - 1 stops the program (#483)
function main() {
    print("printed before");
    const a: number[] = [1, 2, 3];
    a.length = 4294967296 * 1024;
    print("length: ", a.length);
}
