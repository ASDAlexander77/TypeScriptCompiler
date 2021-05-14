enum En {
    A,
    B,
    C,
    D = 4200,
    E,
}

enum En2 {
    D0 = En.D,
    D1,
    D2 = 1,
}

function main() {
    const a = En.A;
    let b: En = En.D;
    print(a, En.B, En.C, En.D);
    print(En2.D0, En2.D1, En2.D2);
    print(b);
}

