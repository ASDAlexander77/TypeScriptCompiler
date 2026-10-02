// -mm=own rejects: `c` is one block, an object literal's, stored into two records, and neither
// record owns it alone. Its `ts.New` is a `value_ref`, a type that owns nothing, so a rule that
// reads the type alone takes it for data nobody owns.
function main() {
    const c = {
        toString() {
            return "Hi";
        },
    };
    const r1 = { a: 1, c: c };
    const r2 = { a: 2, c };
    print(r1.a, r2.a);
}
