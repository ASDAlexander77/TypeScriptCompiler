// the `let` borrows the parameter's local, which is given a new array while the `let` is still used
function f(results: number[] | undefined) { results ??= []; let a = <number[]>results; results = [7];
    a.push(1); return a.length + results.length; }
function main() { print(f(undefined)); }
