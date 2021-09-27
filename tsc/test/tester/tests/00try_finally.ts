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

function main() {
    main1();
    print("done.");
}