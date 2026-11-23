// TSLANG_ISSUES 112: shapes with the same field names whose fields differ in type (`label: string` vs
// `label: object`) were merged into one shape stored as neither ("unrealized_conversion_cast"); they are a union
// told apart by its tag. Not under -mm=own: it can't return an object field read through such a union (a value
// merged from branches) or take an object literal without an optional field into it.
function label(options: { label: string; alias: string } | { label: object; alias?: string }) {
    if (typeof options.label === "string") return options.alias ?? options.label;
    return options.label;
}

function main() {
    assert(label({ label: "x" + 1, alias: "y" + 2 }) == "y2", "string label, alias");
    const o = <object>{};
    assert(label({ label: o }) === o, "object label");
    print("done.");
}
