// -mm=own: a class read by a string key gives `any`, which own does not allow (a keyof key reads the member's
// own type instead)
class C {
    a = 1;
}

function key(s: string) {
    return s;
}

function main() {
    const c = new C();
    print(<number>c[key("a")]);
}
