// -mm=own rejects: `local` borrows the global's value, but a call may store another value into
// the global - and destroy the one `local` reads - while `local` is still used.
let names: string[] = ["a", "b"];
function churn() {
    names = ["c"];
}
function main() {
    let local = names;
    churn();
    print(local.length);
}
