// TSLANG 199: an enum indexed by a key known at run time ("Enum reverse lookup by index is not supported"): a number
// gives the name of the member with that value (the reverse mapping of a numeric enum, "" when none has it), a string
// the value of the member with that name (0 when none has it)
enum Size {
    Small,
    Medium,
    Large
}

enum Mixed {
    A = 1,
    B = 5
}

function nameOf(s: Size) {
    return Size[s];
}

function main() {
    const mySize = Size.Large;
    assert(Size[mySize] == "Large" && nameOf(Size.Small) == "Small" && Size[1] == "Medium", "the reverse mapping");
    assert(Mixed[5] == "B" && Mixed[2] == "", "the reverse mapping of set values");

    const keys = ["Medium", "Large", "Nope"];
    let total = 0;
    for (const k of keys) {
        total += Size[k];
    }

    assert(total == 3, "a member by a name known at run time");
    assert(Size["Large"] == 2, "a member by a constant name");
    print("done.");
}
