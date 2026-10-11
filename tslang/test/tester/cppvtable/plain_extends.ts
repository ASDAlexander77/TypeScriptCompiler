// An ordinary class extending a @cppvtable class would add slots C++ doesn't have.
@cppvtable
class Base {
    first(): s32;
}

class Plain extends Base {
    second(): s32 { return 2; }
}

function main() {
    print("done.");
}
