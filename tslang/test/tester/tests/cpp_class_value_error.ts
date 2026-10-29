// A @cpp class whose objects C++ makes has no `.new` or constructor here: it can't be a value.
@cpp
class Native {
    @slot(0) first(): s32;
}

function main() {
    const k: typeof Native = Native;
    print("done.");
}
