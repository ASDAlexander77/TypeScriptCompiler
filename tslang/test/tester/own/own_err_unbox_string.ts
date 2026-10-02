// -mm=own rejects: `<string>` of an `any` is the box's own string when it holds one, and a new
// string made from its number, boolean or bigint otherwise - a borrow on some paths and a value of
// its own on others.
function main() {
    const s = "value";
    const a = <any>s;
    const back = <string>a;
    print(back);
}
