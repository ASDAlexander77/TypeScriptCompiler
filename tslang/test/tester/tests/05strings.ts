function testStrings(): void {
    print("testStrings");
    assert((<string>42) == "42", "42");

    print("ts0x");
    let s = "live";
    assert(s == "live", "hello eq");
    print("ts0y");

    s = s + "4OK";
    let s2 = s;
    print("ts0");
    //assert(s.charCodeAt(4) == 52, "hello eq2");
    //assert(s.charAt(4) == "4", "hello eq2X");
    assert(s[4] == "4", "hello eq2X");
    assert(s.length == 7, "len7");
    print("ts0");
    s = "";

    //pause(3)
    for (let i = 0; i < 10; i++) {
        print("Y");
        s = s + i;
        print(s);
    }
    assert(s == "0123456789", "for");
    let x = 10;
    s = "";
    while (x >= 0) {
        print("X");
        s = s + x;
        x = x - 1;
    }
    assert(s == "109876543210", "while");
    print(s);
    print(s2);

    s2 = "";
    // don't leak ref

    x = 21;
    s = "foo";
    s = `a${x * 2}X${s}X${s}Z`;
    assert(s == "a42XfooXfoo" + "Z", "`");

    print("X" + true);

    assert("X" + true == "Xt" + "rue", "boolStr");
    print("testStrings DONE");
}

function main() {
    testStrings();

    print("done.");
}
