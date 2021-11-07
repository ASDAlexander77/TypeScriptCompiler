function cycle() {
    print("start...1");

    for (let i = 0; i < 1000000; i++) {
        print(`val : ${i}`);
    }

    print("end.");
}

// TODO: Bug here
function cycle_with_func() {
    print("start...2");

    let i = 2;

    let f = () => { return i; };

    for (let i = 0; i < 1000000; i++) {
        let r = f() + i;
        print(`val : ${r}`);
    }

    print("end.");

}

function main() {
    cycle();
    //cycle_with_func();
    print("done.");
}
