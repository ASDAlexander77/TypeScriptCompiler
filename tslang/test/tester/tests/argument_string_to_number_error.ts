// A string given to a number parameter is an error, as in TypeScript (TS2345). It was converted:
// "7" became 7, and "seven" 0.
function twice(n: number) {
    return n * 2;
}

function main() {
    print(twice("7"));
}
