// "no" | "yes" allows only its two literals: another literal is an error, as in TypeScript (TS2322)
function answer(v: number) {
    return v != 0.0 ? "yes" : "no";
}

function main() {
    let a = answer(1.0);
    a = "maybe";
    print(a);
}
