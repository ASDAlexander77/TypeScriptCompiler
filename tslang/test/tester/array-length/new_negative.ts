// a new array of a negative length stops the program (#483)
type Numbers = number[];

function main() {
    print("printed before");
    let n = -1;
    const a = new Numbers(n);
    print("length: ", a.length);
}
