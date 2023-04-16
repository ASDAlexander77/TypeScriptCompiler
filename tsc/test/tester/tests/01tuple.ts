function main() {

    const obj: { name: string } = { name: 1 };
    print(obj.name);

    assert(obj.name === "1");

    const obj2 = { name: "aa1" };
    print(obj2.name);

    const obj3: { name: string[] } = { name: ["aa1", "asd2"] };
    print(obj3.name[0], obj3.name[1]);

    const obj4: { name: string[] } = { name: [2, 4] };
    print(obj4.name[0], obj4.name[1]);

    assert(obj4.name[0] === "2");
    assert(obj4.name[1] === "4");

    const obj5: { name: { val1: string }[] } = { name: [ { val1: "sss1" }, { val1: "sss2" } ] };
    print(obj5.name[0].val1, obj5.name[1].val1);

    const obj6: { name: { val1: string }[] } = { name: [ { val1: 3 }, { val1: 6 } ] };
    print(obj6.name[0].val1, obj6.name[1].val1);

    assert(obj6.name[0].val1 === "3");
    assert(obj6.name[1].val1 === "6");

    print("done.");
}
