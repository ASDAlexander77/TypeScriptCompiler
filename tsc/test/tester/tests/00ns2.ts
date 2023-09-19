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

enum En {
    A,
    B,
    C,
    D = 4200,
    E,
}

function switchA(e: En) {
    let r = 12;
    switch (e) {
        case En.A:
        case En.B:
            return 7;
        case En.D:
            r = 13;
            break;
    }
    return r;
}

namespace nn {
    enum En {
        A,
        B,
        C,
        D = 4200,
        E,
    }

    function switchA(e: En) {
        let r = 12;
        switch (e) {
            case En.A:
            case En.B:
                return 8;
            case En.D:
                r = 13;
                break;
        }
        return r;
    }
}

function main() {
    ff();
    nn.fff();

    assert(switchA(En.A) == 7, "s1");
    assert(nn.switchA(nn.En.A) == 8, "s2");

    print("done.");
}
