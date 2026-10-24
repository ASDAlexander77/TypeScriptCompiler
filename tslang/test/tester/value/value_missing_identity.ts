// @strict-null false
// Two @value values that may be missing have no identity to compare either: only null and
// undefined are compared with one.
@value
interface Hit {
    readonly dist: number;
}

function find(d: number): Hit | null {
    return { dist: d };
}

function main() {
    print(find(1) === find(1));
}
