// -mm=own: a parameter given a default by `??=` or `||=`, then copied into a `let` through a cast out
// of its optional type (`let a = <number[]>results`). The cast is a merge of the parameter's value and
// an empty default; the `let` borrows the parameter's local, which owns what `??=` made, or borrows
// the caller's array.
class C {
    n = 1;
}

function nullish(results: number[] | undefined) {
    results ??= [];
    let a = <number[]>results;
    a.push(1);
    return a.length;
}

function orElse(results: number[] | undefined) {
    results ||= [];
    let a = <number[]>results;
    a.push(1);
    return a.length;
}

function object(c: C | undefined) {
    c ??= new C();
    let a = <C>c;
    a.n++;
    return a.n;
}

function main() {
    let total = 0;
    for (let i = 0; i < 1000; i++) {
        total += nullish(undefined) + orElse(undefined);
        const keep = [i];
        total += nullish(keep) + orElse(keep) + keep.length;
        total += object(undefined) + object(new C());
    }

    // an array is passed as its {data, length}: the caller's `keep` still has one element
    assert(total == 1000 * (1 + 1 + 2 + 2 + 1 + 2 + 2), "lengths and counts");
    print("done.");
}
