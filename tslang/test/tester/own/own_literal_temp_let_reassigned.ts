// -mm=own (user ruling 11-23): a `let` given a string element of an array literal no local holds holds
// a copy of it, so it may be given another such element later, in a loop too. Borrowing the literal's
// element, it could not ("cannot be assigned"): own accepted that only while the literal leaked (fuzz
// 1122043).
@noinline function ms(i: number): string {
    return "m" + i;
}

function churn() {
    const junk: string[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push("junk" + i + "-" + i);
    }
}

function main() {
    let total: number = 0;
    let last: string = "";
    for (let i = 0; i < 20; i++) {
        let st: string = ["s" + i, ms(i + 7)][1];
        for (let j = 0; j < 3; j++) {
            st = [(j % 2 == 0 ? ms(j + 5) : "k" + j), st + j][1];
            churn();
        }

        total += st.length;
        last = st;
    }

    print(total, last);
    assert(last == "m26012", "last");
    print("done.");
}
