// Borrow facts and a -shared import: this `sumOf` comes from the DLL facts_lib in the working
// directory, so the static library's fact for a function of that name must not apply.

@static
import { sumOf } from "./facts_lib";

function call(n: number) {
    const a = [n, n];
    return sumOf(a);
}

function main() {
    print(call(1));
}
