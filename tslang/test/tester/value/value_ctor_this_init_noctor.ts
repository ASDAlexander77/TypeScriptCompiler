// ...with no constructor written as well (`show(this)` read a = 0, not 7)
function show(v: V) {
    return v.a;
}

@value
class V {
    readonly b: number = show(this);
    readonly a: number = 7;
}

function main() {
    print(new V().b);
}
