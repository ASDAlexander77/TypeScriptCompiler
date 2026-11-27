// TSLANG 194: a field some members of a union of shapes have and another has not is an error, as in TypeScript
type Validated = { valid: boolean } | { valid: boolean; msg: string };

function get(b: boolean): Validated {
    if (b) {
        return { valid: true };
    }

    return { valid: false, msg: "m" };
}

function main() {
    const v = get(true);
    print(v.msg);
}
