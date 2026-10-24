// A @value type is the tuple of its fields, held in place: one that holds itself, even as
// `L | null`, would be infinitely large.
@value
interface L {
    readonly v: number;
    readonly next: L | null;
}

function main() {
    const a: L = { v: 1, next: null };
    const b: L = { v: 2, next: a };
    print(b.next ? b.next.v : -1);
}
