// @cppnontrivial says how a returned C++ record is returned: on a method returning a number it is
// a mistake.
@cpp
class Methods {
    @cppnontrivial
    @slot(0) count(): s32;
}

function main() {
    print("done.");
}
