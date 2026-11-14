// a boolean given to a string is an error (TS2322): converted, `s = flag` made s "true"
function main() {
    const flag: boolean = true;
    let s: string = flag;
    print(s);
    print("done.");
}
