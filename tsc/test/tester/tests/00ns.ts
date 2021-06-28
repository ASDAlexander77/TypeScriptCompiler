namespace nn {
    function ff() {
        print("hello");
    }

    function fff() {
        ff();
    }
}

function f() {
    print("hello");
}

function main() {
    f();
    nn.fff();
}
