// -mm=own rejects: `greet` returns its argument on one path and a new string on the other, so
// its callers cannot be told whether they own the result.
function greet(name: string) {
    if (name === "Honda") return name;
    return "Sorry, " + name;
}
function main() {
    print(greet("Honda"));
}
