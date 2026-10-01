// A length is an index; it holds, compares and prints as a signed value, as a JS number would:
// a variable initialised from one keeps -1 when assigned it, and a length minus more is negative
function assigned(s: string) {
    let n = s.length;
    n = -1;
    assert(n == -1, "n == -1");
    assert(n < 0, "n < 0");
    const asNumber: number = n;
    assert(asNumber == -1, "as number");
    assert(`${n}` == "-1", "as text");
    assert("" + n == "-1", "concatenated");
    const boxed: any = n;
    assert(`${boxed}` == "-1", "boxed");
}

function subtracted(s: string, a: number[]) {
    assert(`${s.length - 3}` == "-1", "string length - 3");
    assert(`${a.length - 5}` == "-3", "array length - 5");
    assert(a.length - 5 < 0, "array length - 5 < 0");

    let i = s.length;
    i--; i--; i--; i--;
    assert(`${i}` == "-2", "decremented");

    let w = a.length;
    w -= 5;
    assert(`${w}` == "-3", "-=");
}

function countdown(s: string) {
    let visited = 0;
    for (let k = s.length - 1; k >= 0; k--) {
        visited++;
        if (visited > 10) break;
    }

    assert(visited == 2, "countdown");
}

function main() {
    assigned("ab");
    subtracted("ab", [1, 2]);
    countdown("ab");
    print("done.");
}
