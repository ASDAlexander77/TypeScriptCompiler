// ...nor an element of an array
@value
class V {
    readonly a: number;

    constructor(arr: V[]) {
        arr.push(this);
        this.a = 1;
    }
}

function main() {
    const arr: V[] = [];
    const v = new V(arr);
    print(arr[0].a, v.a);
}
