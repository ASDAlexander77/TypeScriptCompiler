// @strict-null false
// A @value type has no null, even where null checks are not strict: a missing value is written
// `T | null`.
@value
interface Hit {
    readonly dist: number;
}

function f(): Hit {
    return null;
}

function main() {
    print(f().dist);
}
