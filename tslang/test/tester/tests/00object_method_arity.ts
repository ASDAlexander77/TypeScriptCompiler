// An object-literal method given as an interface's function member of another arity: next() for a member
// next(value?) (the value ignored, as JavaScript does), next(v?) for next() (v undefined). Was
// "can't assign '() => ...' to '(p1: number | undefined) => ...': it takes 0 argument(s), 2 expected".
type R = { value: number, done: boolean };
interface WithArgField {
    next: (value?: number) => R;
}
interface WithArgMethod {
    next(value?: number): R;
}
interface NoArgField {
    next: () => R;
}

function main() {
    const a: WithArgField = { next() { return { value: 1, done: false }; } };
    print(a.next(3).value);
    const b: WithArgMethod = { next() { return { value: 2, done: false }; } };
    print(b.next(3).value);
    const c: NoArgField = { next(v?: number) { return { value: v === undefined ? 30 : v, done: false }; } };
    print(c.next().value);
    print("done.");
}
