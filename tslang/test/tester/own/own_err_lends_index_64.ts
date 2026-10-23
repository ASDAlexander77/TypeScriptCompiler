// -mm=own rejects: a method's '@lends' is kept as a mask of 64 bits for the class's declaration,
// so it can name only the method's arguments 0 to 63 (the receiver is 0); 'p63' is argument 64.
class V {
    constructor(public x: number) {}
}

class K {
    @lends("p63")
    pick(
        p0: V, p1: V, p2: V, p3: V, p4: V, p5: V, p6: V, p7: V,
        p8: V, p9: V, p10: V, p11: V, p12: V, p13: V, p14: V, p15: V,
        p16: V, p17: V, p18: V, p19: V, p20: V, p21: V, p22: V, p23: V,
        p24: V, p25: V, p26: V, p27: V, p28: V, p29: V, p30: V, p31: V,
        p32: V, p33: V, p34: V, p35: V, p36: V, p37: V, p38: V, p39: V,
        p40: V, p41: V, p42: V, p43: V, p44: V, p45: V, p46: V, p47: V,
        p48: V, p49: V, p50: V, p51: V, p52: V, p53: V, p54: V, p55: V,
        p56: V, p57: V, p58: V, p59: V, p60: V, p61: V, p62: V, p63: V
    ): V {
        return p63;
    }
}

function main() {
    print("done.");
}
