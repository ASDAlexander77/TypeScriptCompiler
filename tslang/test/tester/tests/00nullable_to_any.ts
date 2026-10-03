// A nullable class, string or array (a union with null and no tag: one pointer) boxed into `any`
// and read back (#455). Boxing it crashed the compiler: the union itself has no `typeof` name, so
// the box had no type descriptor.
class Node {
    v = 5;
}

function nodeOrNull(set: boolean): Node | null {
    if (set) return new Node();
    return null;
}

function main() {
    const n: Node | null = nodeOrNull(true);
    const boxed: any = n;
    const back = <Node>boxed;
    assert(back.v == 5, "a class boxed from a nullable union comes back");

    let s: string | null = "text" + 1;
    const boxedString: any = s;
    assert(<string>boxedString == "text1", "a string boxed from a nullable union comes back");

    const a: number[] | null = [1, 2, 3];
    const boxedArray: any = a;
    assert((<number[]>boxedArray).length == 3, "an array boxed from a nullable union comes back");

    const none: Node | null = nodeOrNull(false);
    const boxedNone: any = none;
    assert(boxedNone == null, "null boxed from a nullable union is null");

    print("done.");
}
