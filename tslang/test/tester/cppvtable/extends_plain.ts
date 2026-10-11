// A @cppvtable class extending an ordinary class would have the ordinary class's slots first.
class Plain {
    m(): s32 { return 1; }
}

@cppvtable
class Base extends Plain {
    first(): s32;
}

function main() {
    print("done.");
}
