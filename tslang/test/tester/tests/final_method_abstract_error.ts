// TSLANG_ISSUES 48: an abstract method has no body to run, so it cannot be @final
abstract class Base {
    @final abstract factor(): number;
}

function main() { }
