// TSLANG_ISSUES 111: `+` of two `any`s may add numbers or join strings, which is known only at run time
// (it failed in the lowering, with no location)
function main() {
    let x: any = 1;
    let y: any = 2;
    print(x + y);
    print("done.");
}
