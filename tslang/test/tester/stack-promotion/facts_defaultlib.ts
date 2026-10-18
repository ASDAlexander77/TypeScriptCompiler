// Borrow facts from the default library: Math.max's rest array is built here and only read there.
// See borrow-facts-defaultlib.cmake.

function biggest(n: number) {
    return Math.max(n, n * 2, n - 1);
}

function main() {
    print(biggest(3));
}
