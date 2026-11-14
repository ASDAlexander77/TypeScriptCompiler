// a `string` given to a union of string literals is an error, as in TypeScript (TS2322)
type Answer = "no" | "yes";

function main() {
    const s: string = "yes";
    const a: Answer = s;
    print(a);
}
