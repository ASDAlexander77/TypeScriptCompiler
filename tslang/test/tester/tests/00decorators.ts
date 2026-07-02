@used
let a = 10;

@noinline
@optnone
@used
function test1() {
    print("Hello 1");
}

function test2() {
    print("Hello 2");
}

function main() {

    test1();
    test1();

    test2();
    test2();

    print("done.");
}