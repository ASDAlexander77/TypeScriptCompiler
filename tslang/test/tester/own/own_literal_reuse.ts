// -mm=own: literals repeated through a function. Under --opt, CSE merges identical literal casts
// before ownership inference, so one SSA value is stored into several places. A number owns
// nothing, and a string literal is the immortal global, which any number of places may hold -
// neither is moved by being stored.
function count(n: number) {
    let t = 0;
    let u = 2;
    for (let i = 0; i < n; i++) {
        t = 0;
        t += i;
        if (i % 3 == 0) {
            u = 0;
        }
    }

    return t + u;
}

function main() {
    let s = "abc";
    s = "abc";
    let q = "abc";
    assert(count(5) == 4);
    assert(s.length + q.length == 6);
    print("done.");
}
