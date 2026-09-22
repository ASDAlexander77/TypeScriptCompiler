function main() {
    let f: ((a: number, b: number) => void) | undefined = (a: number, b: number) => { print(a + b); };
    f?.(1);
}
