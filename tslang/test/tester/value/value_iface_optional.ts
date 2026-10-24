// A field of a @value interface is always there: an absent field has no slot in the value.
@value
interface R {
    readonly f?: number;
}

function main() {
}
