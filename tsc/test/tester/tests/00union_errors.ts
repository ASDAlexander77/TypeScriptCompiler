type TokenType = "NUMBER" | "PLUS" | "MINUS" | "MULTIPLY" | "DIVIDE" | "LEFT_PAREN" | "RIGHT_PAREN" | "EOF";

let v: TokenType = "PLUS";
let v2: TokenType = "DIVIDE";

assert(v === "PLUS")
assert(v !== "MINUS")

assert(v === v)
assert(v !== v2)


let a: number | null = 1.0;

if (a !== null) {
    let r = a + 1;
    print (r);
    assert(r == 2.0);
}

print("done.")