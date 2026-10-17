// A number on the right of `instanceof` is no class: the test compared the two types and
// was quietly true or false.

type str = TypeOf<"">;

function main() {
    print("asd" instanceof str);
    print(5 instanceof 3);
}
