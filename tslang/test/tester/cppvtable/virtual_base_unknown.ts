// @virtual numbers a class's methods after its base class's slots, so the base class's size has to
// be known: a base naming only some of its slots with @slot(N) can't say where they end.
@cpp
class Partial {
    @slot(3) known(): s32;
}

@cpp
class Extended extends Partial {
    @virtual more(): s32;
}

function main() {
}
