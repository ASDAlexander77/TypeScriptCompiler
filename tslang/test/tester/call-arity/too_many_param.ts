function call(f: (a: number) => void) {
    f(1, 2);
}

function main() {
    call((a: number) => { print(a); });
}
