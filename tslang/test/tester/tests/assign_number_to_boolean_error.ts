// a number given to a boolean is an error (TS2322): converted, `flag = 5` made flag true
function main() {
    let flag: boolean = false;
    const n: number = 5;
    flag = n;
    print(flag);
    print("done.");
}
