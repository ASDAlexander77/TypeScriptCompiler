// A @cpp class extending an ordinary class would have the ordinary class's slots first.
class Plain {
    m(): s32 { return 1; }
}

@cpp
class Base extends Plain {
    @slot(0) first(): s32;
}

function main() {
    print("done.");
}
