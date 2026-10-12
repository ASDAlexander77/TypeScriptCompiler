// A method overriding a base class's slot keeps that slot: @slot naming another is an error.
@cpp
class Base {
    @slot(0) first(): s32;
}

@cpp
class Derived extends Base {
    @slot(2) first(): s32;
}

function main() {
}
