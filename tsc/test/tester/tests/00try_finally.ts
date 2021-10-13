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

function try2() {
    let i = 10;
    try {
        glb1++;
        print("Try");
        if (i > 0)
            throw 1.0;
        print("cont");
    }
    finally {
        glb1++;
        print("Finally");
    }
}

function main2() {
    glb1 = 0;
    try {
        try2();
    }
    catch
    {
        glb1++;
        print("caught");
    }

    assert(glb1 == 3);
}


function main3() {

    let c = 0;

    try {
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
    }
    catch (e: string) {
        c++;
    }

    assert(3 == c);
}

function main4() {

    let i = 0;

    try {
        try {
            throw 1.0;
        }
        catch (e: number) {
            i++;
            print("asd1");
            throw 2.0;
        }
        finally {
            print("finally");
        }
    }
    catch (e2: number) {
        i++;
        print("asd3");
    }

    assert(i == 2);
}

function main() {
    main1();
    main2();
    main3();
    main4();
    print("done.");
}