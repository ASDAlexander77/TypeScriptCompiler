// None of these drops a fraction it was not asked to drop: no warning.
function quiet() {
    let n = 0;
    n += 1;
    n = n * 2;
    n = 3.0;
    let declared: s32 = 0;
    declared = 2.5;
    let f = 0.0;
    f += 1.5;
    let typed: number = 0;
    typed += 1.5;
    return n + declared + f + typed;
}

print(quiet());
