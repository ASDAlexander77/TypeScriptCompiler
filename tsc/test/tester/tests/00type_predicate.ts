function s(v: any): v is string {
    return typeof v === "string";
}

function main() {
    assert(s("sss"));
    print("done.")
}