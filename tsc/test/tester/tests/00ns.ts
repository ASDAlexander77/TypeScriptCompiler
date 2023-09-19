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

namespace nn1 {
    type Type1 = TypeOf<1>;
}

function f1(p: nn1.Type1) {}

function main() {
    f();
    nn.fff();
    f1(10);
    print("done.");
}
