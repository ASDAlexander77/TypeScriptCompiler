// A `switch` on a char with string cases casts the char to a string once per case. That cast
// allocates, and CastOp reported no effect for it, so CSE merged the casts into one: under -mm=rc
// the first case's release freed the string that every later case then retained, compared and
// released again - a use after free and a double free for each character that missed the first
// case. RegExp's constructor reads its flags this way.
function flagsOf(flags: string) {
    let g = 0;
    let i = 0;
    let m = 0;
    let other = 0;
    for (const c of flags) {
        switch (c) {
            case "s":
                break;
            case "g":
                g++;
                break;
            case "i":
                i++;
                break;
            case "m":
                m++;
                break;
            default:
                other++;
                break;
        }
    }

    return g * 1000 + i * 100 + m * 10 + other;
}

function main() {
    let t = 0;
    for (let n = 0; n < 10000; n++) {
        t += flagsOf("gimxgim");
    }

    assert(t == 10000 * 2221);
    print("done.");
}
