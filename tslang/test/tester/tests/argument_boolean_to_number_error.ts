// a boolean given for a number parameter is an error (TS2345)
function twice(n: number): number {
    return n * 2;
}

function main() {
    const flag: boolean = true;
    print(twice(flag));
    print("done.");
}
