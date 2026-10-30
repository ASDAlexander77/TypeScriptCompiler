// One string pushed twice in one push: rc's two retains for it each copy a different operand, the
// first after the retain that is still the string (copyForRetain takes the nearest use when there are
// several, spec 71). It was rejected as a use after a move.
function churn() {
    const hold: string[] = [];
    for (let i = 0; i < 64; i++) {
        hold.push("churn" + i);
    }
}

function main() {
    const s = "s" + 1;
    const list: string[] = [];
    list.push(s, s);
    churn();
    assert(list[0] == "s1" && list[1] == "s1", "two copies");
    assert(s == "s1", "the string itself");
    print(list[0], s);
    print("done.");
}
