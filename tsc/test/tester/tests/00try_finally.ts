function main1() {

    let c = 0;

    try {
        c++;
        print("try");
    } finally {
        c++;
        print("finally");
    }

    assert(2 == c);
}

function main2() {

    let c = 0;

    try {
        c++;
        print("try");
        throw "except";
        c--;
        print("after catch");
    } finally {
        c++;
        print("finally");
    }

    assert(2 == c);
}

function main() {
    main1();
    main2();
    print("done.");
}