// A class numbers its new virtual methods with @virtual or with @slot(N), not both: a @slot among
// @virtual methods would shift or collide with them.
@cpp
class Mixed {
    @virtual first(): s32;
    @slot(5) second(): s32;
}

function main() {
}
