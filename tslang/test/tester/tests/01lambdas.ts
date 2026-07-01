function reduce2<T, V = T>(this: T[], func: (v: V, t: T) => V, initial?: V) {
    let result = initial || 0;
    for (const v of this) result = func(result, v);
    return result;
}

function main() {

    const array1 = [1, 2, 3, 4];
  
    // 0 + 1 + 2 + 3 + 4
    const initialValue = 0;

    const sumWithInitial2 = array1.reduce2(
      (accumulator: int, currentValue) => accumulator + currentValue,
    );
  
    print (sumWithInitial2);

    assert(sumWithInitial2 == 10);

    print ("done.");
}
  