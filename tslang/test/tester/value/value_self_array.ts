// ...or an array of itself: the array's element type is the type being laid out
@value
interface L {
    readonly v: number;
    readonly next: L[];
}

function main() {
    const a: L = { v: 1, next: [] };
    print(a.v);
}
