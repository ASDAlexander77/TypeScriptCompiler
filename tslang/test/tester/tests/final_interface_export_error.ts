// A @final interface is not exported: another module could implement it.
@final
export interface Shape {
    area(): number;
}

function main() {
    print("done.");
}
