// A namespace whose function names a class declared after it is generated again once the class is
// there; the functions the first try had generated were generated twice ("redefinition of symbol").
namespace Forward {
    export function first(x: number) {
        return x + 1;
    }

    export function second(x: number) {
        return x == Limits.BIG ? 0 : 1;
    }
}

class Limits {
    public static BIG: number = 5;
}

function main() {
    assert(Forward.first(1) == 2, "the function generated on the first try");
    assert(Forward.second(5) == 0 && Forward.second(2) == 1, "the function naming the later class");

    print("done.");
}
