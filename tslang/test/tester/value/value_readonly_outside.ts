// The fields of a @value value are readonly: a @value class's value held in a variable is not
// assigned (a @value interface's: value_readonly_outside_iface.ts).
@value
class V {
    constructor(public readonly x: number) {}
}

function main() {
    const v = new V(1);
    v.x = 2;
    print(v.x);
}
