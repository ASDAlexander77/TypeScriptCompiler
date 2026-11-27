// TSLANG 140: a declared class's method binds one external symbol, so its overloads are an error (they were
// "redefinition of symbol")
declare class DC {
    m(x: number): number;
    m(x: string): string;
}

function main() {
    print("done.");
}
