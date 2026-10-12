// An ordinary class extending a @cpp class would add slots C++ doesn't have.
@cpp
class Base {
    @slot(0) first(): s32;
}

class Plain extends Base {
    second(): s32 { return 2; }
}

function main() {
    print("done.");
}
