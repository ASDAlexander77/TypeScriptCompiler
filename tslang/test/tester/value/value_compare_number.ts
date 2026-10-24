// @strict-null false
// A @value value is compared with null or undefined only: against a number it has no identity to
// compare either.
@value
interface Hit {
    readonly dist: number;
}

function find(d: number): Hit | null {
    return { dist: d };
}

function main() {
    print(find(1) === 1);
}
