// A function that returns a borrow of its argument on one path and a new string on another: the
// borrowed return is copied, so the result is always the caller's (spec 22.5). Also the
// synthesized `<string>` of an `any`, and a union cast to a string.
function greet(name: string) {
    if (name === "Honda") return name;
    return "Sorry, " + name;
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    let who = "Hon" + "da";
    const g = greet(who);
    who = "x" + 1;
    assert(churn() == 1000);
    assert(g == "Honda");
    assert(greet("Bob") == "Sorry, Bob");

    const a = <any>("boxed" + 1);
    const back = <string>a;
    assert(back == "boxed1");

    let u: number | string = "un" + 1;
    const s = <string>u;
    u = 3;
    assert(churn() == 1000);
    assert(s == "un1");

    print("done.");
}
