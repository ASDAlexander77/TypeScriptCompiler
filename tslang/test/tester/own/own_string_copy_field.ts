// A field's string given a second owner is copied (spec 22): stored into another field, returned,
// and pushed. Each original is then overwritten, or its owner released, and its memory reused before
// the copy is read.
class Rec {
    str = "";
    str2 = "";
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

// returns a field of an object it releases: not a borrow of a parameter (spec 22.4)
function textOf(n: number) {
    const made = new Rec();
    made.str2 = "made" + n;
    return made.str2;
}

function main() {
    const r = new Rec();
    r.str2 = "two" + 2;
    r.str = r.str2;
    r.str2 = "other" + 3;
    assert(churn() == 1000);
    assert(r.str == "two2" && r.str2 == "other3");

    const got = textOf(4);
    assert(churn() == 1000);
    assert(got == "made4");

    const list: string[] = [];
    list.push(r.str2);
    r.str2 = "last" + 5;
    assert(churn() == 1000);
    assert(list[0] == "other3");

    print("done.");
}
