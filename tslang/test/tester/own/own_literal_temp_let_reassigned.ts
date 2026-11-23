// -mm=own (user ruling 11-23): a `let` given a string element of an array literal no local holds holds
// a copy of it, so it may be given another such element later, in a loop too. Borrowing the literal's
// element, it could not ("cannot be assigned"): own accepted that only while the literal leaked (fuzz
// 1122043). A `let` of a wider type (`string | null`, a union) is given the copy through the cast that
// widens it.
@noinline function ms(i: number): string {
    return "m" + i;
}

function churn() {
    const junk: string[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push("junk" + i + "-" + i);
    }
}

function exact(): string {
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
    return last;
}

function nullable(): string {
    let last: string = "";
    for (let i = 0; i < 20; i++) {
        let st: string | null = ["s" + i, ms(i + 7)][1];
        for (let j = 0; j < 3; j++) {
            st = [ms(j + i), "a" + j][0];
            churn();
        }

        if (st !== null) {
            last = st;
        }
    }

    print(last);
    return last;
}

function union(): string {
    let text = "";
    for (let i = 0; i < 20; i++) {
        let st: string | number = ["s" + i, ms(i + 7)][1];
        churn();
        text = "" + st;
        for (let j = 0; j < 3; j++) {
            st = [ms(j * i), "b" + j][0];
            churn();
        }

        text = text + "/" + st;
    }

    print(text);
    return text;
}

function main() {
    assert(exact() == "m26012", "exact");
    assert(nullable() == "m21", "nullable");
    assert(union() == "m26/m38", "union");
    print("done.");
}
