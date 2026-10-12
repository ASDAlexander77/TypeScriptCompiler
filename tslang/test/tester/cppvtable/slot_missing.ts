// A method of a @cpp class without @slot isn't virtual, and one without a body is then a C++
// non-virtual method, which needs its symbol: a forgotten @slot is an error, not a wrong call.
@cpp
class Base {
    @slot(0) first(): s32;
    second(): s32;
}

function main() {
}
