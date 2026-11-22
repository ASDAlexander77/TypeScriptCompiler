interface Source {
    next: (value?: number) => number;
}
interface Target {
    next: () => number;
}
function take(t: Target) { return t.next(); }
function main() {
    const s: Source = { next: (value?: number) => 1 };
    take(s);
}
