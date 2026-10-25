// a @cpp class's method whose @cppname doesn't parse: reported, not taken for a method without
// a symbol
@cpp
class Methods {
    @cppname("int scaled(int")
    scaled(x: s32): s32;
}

function main() {
}
