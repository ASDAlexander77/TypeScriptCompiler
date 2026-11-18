// an `any` cast to an object type of 676 x 676 fields: the type's name went into the check made for the cast
type props = "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z";
type manyprops = `${props}${props}`;
const c = [null as any as {[K in manyprops]: {[K2 in manyprops]: `${K}.${K2}`}}][0];
