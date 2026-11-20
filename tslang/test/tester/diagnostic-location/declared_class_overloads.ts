// A declared class's members are external symbols, one each: overloads of one are an error at the second, as a
// declared function's are ("redefinition of symbol named 'C.constructor'" from the verifier).
declare class C {
    constructor(n: number);
    constructor(s: string);
}
