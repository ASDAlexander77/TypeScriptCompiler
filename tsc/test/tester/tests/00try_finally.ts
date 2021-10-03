let glb1 = 0;
let x = 0;

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

function pause(ms: number): void {
    print("pause ms", ms);
}

function throwVal(n: number) {
    pause(1)
    if (n > 0)
        throw n
    pause(1)
}

function callingThrowVal(k: number) {
    try {
        pause(1)
        throwVal(k)
        pause(1)
        glb1++
    } catch (e: number) {
        print("catch", e);
        assert(e == k)
        glb1 += 10
        if (k >= 10)
            throw e
    } finally {
        x += glb1
    }
}

function main3() {
    print("test exn")
    glb1 = x = 0
    callingThrowVal(1)
    assert(glb1 == 10 && x == 10)
    callingThrowVal(0)
    assert(glb1 == 11 && x == 21)
    callingThrowVal(3)
    assert(glb1 == 21 && x == 42)
    print("test exn done")
}

function main() {
    main1();
    main2();
    main3();
    print("done.");
}