// A case of another kind of primitive than the switch's could never match: the number was made
// a string to compare, with no error.

function main() {
    const n = 5;
    switch (n) {
        case 5: print("five"); break;
        case "a": print("a"); break;
    }
}
