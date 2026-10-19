// Y fits F member by member, but F is @final: a call on an F is a direct call to F's method, so a
// Y passed off as an F would run F's body on a Y's fields.
class Y {
    m(): int {
        return 7;
    }
}

@final
class F {
    m(): int {
        return 2;
    }
}

function main() {
    const f: F = new Y();
    print(f.m());
}
