// TSLANG_ISSUES 110: `new` on a plain function (a constructor function) is refused with an error that says so;
// it failed verification ("'ts.New' op result #0 must be ...")
function Point(x: number) {
}

function main() {
    const p = new Point(3);
    print("done.");
}
