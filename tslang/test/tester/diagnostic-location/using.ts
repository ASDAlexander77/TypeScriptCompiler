// A `using` of a value with no dispose method was reported at the start of the file: the dispose
// call the compiler makes for it had no position.

function main() {
    using x = 5;
}
