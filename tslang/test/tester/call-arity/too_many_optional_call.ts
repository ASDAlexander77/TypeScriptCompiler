class O {
    m(a: number, b: number) {
        return a + b;
    }
}

function main() {
    const o: O | undefined = new O();
    print(o?.m(1, 2, 3));
}
