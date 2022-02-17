class Cat { name = "kitty"; meow() { print("meow"); } }
class Dog { name = "doggy"; bark() {} }

function isCat(a: Cat | Dog): a is Cat {
    return a.name === "kitty";
}

function main() {
    let x: Cat | Dog = new Cat();
    if (isCat(x)) {
        x.meow(); // OK, x is Cat in this block
    }

    print("done.");
}