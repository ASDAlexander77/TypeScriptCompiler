// A value with no fields of its own passed where an interface with members is expected compiled
// without a word, and the callee read garbage. It is an error, as in TypeScript.
interface IKind {
    kind: number;
}

function f(i: IKind) {
    return i.kind;
}

print(f(5));
