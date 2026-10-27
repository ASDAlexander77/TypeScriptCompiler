// -mm=own rejects: a generator's state that owns its arguments owns every one it holds, so `@keeps`
// has to name the callback too.
type It = { next(): { value: any, done: boolean } };

function* count(n: number) {
    for (let i = 0; i < n; i++) yield i;
}

@keeps("iter")
function* mapped<TIter extends It>(iter: TIter, f: (x: number) => number) {
    for (const v of iter) yield f(v);
}

function main() {
    for (const v of mapped(count(3), (x) => x * 2)) print(v);
}
