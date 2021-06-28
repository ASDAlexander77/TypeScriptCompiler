namespace nn {
    function ff() {
        print("hello", En1.V1);
    }

    function fff() {
        ff();
    }

    enum En1 {
        V1,
    }
}

enum En1 {
    V1,
}

function ff() {
    print("hello", En1.V1);
}

function main() {
    ff();
    nn.fff();
    print("done.");
}
