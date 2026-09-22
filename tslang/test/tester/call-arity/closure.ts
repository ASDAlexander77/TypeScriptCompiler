function main() {
    let captured = 10;
    const add = (a: number, b: number) => a + b + captured;
    print(add(1));
}
