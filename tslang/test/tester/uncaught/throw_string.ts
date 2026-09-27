// nothing catches it: the run must end with an error, not a crash of the compiler
function main() {
    print("before");
    throw "not caught";
}
