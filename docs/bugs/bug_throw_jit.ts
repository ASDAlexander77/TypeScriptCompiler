type int = 1;
function main1() {
    print("try/catch 1");

    let t = 1;

    try {
        throw 2;
    } catch (v: int) {
        print("Hello ", v);
        t = v;
    }

    assert(2 == t);
}

function main() {
    main1();
    print("done.");
}
