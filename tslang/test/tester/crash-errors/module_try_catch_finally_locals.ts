// a `let` in both the catch and the finally of a module-level try: the finally's block is an unwind cleanup
// scope, which set the personality of a function there is none of
let n: number;
try {
    let l = 0;
    n = l;
} catch (e) {
    let l = 0;
    n = l;
} finally {
    let l = 0;
    n = l;
}
