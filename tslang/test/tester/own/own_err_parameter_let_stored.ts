// TSLANG_ISSUES 57: a `let` that only holds parameters owns nothing - so storing it into a field
// keeps the caller's value, which an exported function cannot ask its callers to give up.
class C {
    a: int[] | null = null;
}

export function f(a: int[], b: int[], c: C): int {
    let s = a;
    s = b;
    c.a = s;
    return 0;
}

function main() { print(f([1, 2], [3, 4, 5], new C())); }
