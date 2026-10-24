// Two @value values have no identity to compare: each is a copy.
@value
interface Ray {
    readonly start: number;
    readonly dir: number;
}

function main() {
    const r1: Ray = { start: 1, dir: 2 };
    const r2: Ray = { start: 1, dir: 2 };
    print(r1 === r2);
}
