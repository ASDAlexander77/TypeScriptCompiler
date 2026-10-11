// @cppnontrivial says how a returned C++ record is returned: on a method returning a number it is
// a mistake.
@cppvtable
class Methods {
    @cppnontrivial
    count(): s32;
}

function main() {
    print("done.");
}
