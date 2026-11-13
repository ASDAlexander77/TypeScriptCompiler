// ...unless it says how many slots it has, @vtablesize(N): the derived class's @virtual methods
// start there.
@cpp
@vtablesize(6)
class Partial {
    @slot(3) known(): s32;
}

@cpp
class Extended extends Partial {
    @virtual more(): s32;
}

declare function make_extended(): Extended;

function main() {
    print(make_extended().more());
}
