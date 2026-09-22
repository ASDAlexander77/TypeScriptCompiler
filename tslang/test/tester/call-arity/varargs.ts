function va(a: number, ...rest: number[]) {
    return a + rest.length;
}

function main() {
    print(va());
}
