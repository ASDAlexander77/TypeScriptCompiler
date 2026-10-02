// A string moved into a field and used after: the field takes a copy, the local keeps its own.
class H {
    text = "";
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function main() {
    const h = new H();
    let kept = "kept" + 1;
    h.text = kept;
    print(kept);
    h.text = "other" + 2;
    assert(churn() == 1000);
    assert(kept == "kept1" && h.text == "other2");

    print("done.");
}
