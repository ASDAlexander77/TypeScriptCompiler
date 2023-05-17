function toLength(this: string[]) {
    return this.length;
}

function toLength2<T>(this: T[]) {
    return this.length;
}

function toLength3<T, V>(this: T[], v: V) {
    print("v:", v);
    return this.length;
}

function toString<T>(this: T)
{
    return `${this}`;
}

namespace number
{
	function toString1(this: number)
	{
		return `number ${this}`;
	}
}

namespace string
{
	function toString1(this: string)
	{
		return `string ${this}`;
	}
}

function main() {
    let arr = ["asd", "asd2"];
    print(arr.toLength());
    assert(arr.toLength() == 2);

    print(arr.toLength2());
    assert(arr.toLength2() == 2);

    print(arr.toLength3(10));
    assert(arr.toLength3(10) == 2);

    print(arr.toLength3<string, TypeOf<1>>(10));
    assert(arr.toLength3<string, TypeOf<1>>(10) == 2);

    let arrInt = [1, 2];

    print(arrInt.toLength2());
    assert(arrInt.toLength2() == 2);

    print(arrInt.toLength3(10));
    assert(arrInt.toLength3(10) == 2);

    print(arrInt.toLength3<TypeOf<1>, TypeOf<1>>(10));
    assert(arrInt.toLength3<TypeOf<1>, TypeOf<1>>(10) == 2);

    const n = (5).toString();
    print(n);
    assert (n == "5");

    // lookup in namespaces
    print((5.0).toString1());
    print(("asd").toString1());

    assert((5.0).toString1() == "number 5");
    assert(("asd").toString1() == "string asd");

    print("done.");
}