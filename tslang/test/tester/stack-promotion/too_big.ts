// 40 number fields and a vtable pointer: over the 256-byte default, so heap; under a 512 limit,
// stack.
class Big {
    f0: number = 0;
    f1: number = 1;
    f2: number = 2;
    f3: number = 3;
    f4: number = 4;
    f5: number = 5;
    f6: number = 6;
    f7: number = 7;
    f8: number = 8;
    f9: number = 9;
    f10: number = 10;
    f11: number = 11;
    f12: number = 12;
    f13: number = 13;
    f14: number = 14;
    f15: number = 15;
    f16: number = 16;
    f17: number = 17;
    f18: number = 18;
    f19: number = 19;
    f20: number = 20;
    f21: number = 21;
    f22: number = 22;
    f23: number = 23;
    f24: number = 24;
    f25: number = 25;
    f26: number = 26;
    f27: number = 27;
    f28: number = 28;
    f29: number = 29;
    f30: number = 30;
    f31: number = 31;
    f32: number = 32;
    f33: number = 33;
    f34: number = 34;
    f35: number = 35;
    f36: number = 36;
    f37: number = 37;
    f38: number = 38;
    f39: number = 39;
}

function sum() {
    const b = new Big();
    return b.f0 + b.f39;
}

function main() {
    assert(sum() == 39);
    print("done.");
}
