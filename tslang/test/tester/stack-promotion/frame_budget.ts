// 31 number fields and a vtable pointer: 256 bytes each, at the object limit. Each site takes 264
// with its header, so 15 fit the 4096-byte frame budget and the 16th does not.
class Wide {
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
}

function sum() {
    let t: number = 0;
    const o0 = new Wide();
    t += o0.f0;
    const o1 = new Wide();
    t += o1.f1;
    const o2 = new Wide();
    t += o2.f2;
    const o3 = new Wide();
    t += o3.f3;
    const o4 = new Wide();
    t += o4.f4;
    const o5 = new Wide();
    t += o5.f5;
    const o6 = new Wide();
    t += o6.f6;
    const o7 = new Wide();
    t += o7.f7;
    const o8 = new Wide();
    t += o8.f8;
    const o9 = new Wide();
    t += o9.f9;
    const o10 = new Wide();
    t += o10.f10;
    const o11 = new Wide();
    t += o11.f11;
    const o12 = new Wide();
    t += o12.f12;
    const o13 = new Wide();
    t += o13.f13;
    const o14 = new Wide();
    t += o14.f14;
    const o15 = new Wide();
    t += o15.f15;
    const o16 = new Wide();
    t += o16.f16;
    const o17 = new Wide();
    t += o17.f17;
    const o18 = new Wide();
    t += o18.f18;
    const o19 = new Wide();
    t += o19.f19;
    return t;
}

function main() {
    assert(sum() == 190);
    print("done.");
}
