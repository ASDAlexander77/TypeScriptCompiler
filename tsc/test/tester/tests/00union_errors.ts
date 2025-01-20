type TokenType = "NUMBER" | "PLUS" | "MINUS" | "MULTIPLY" | "DIVIDE" | "LEFT_PAREN" | "RIGHT_PAREN" | "EOF";

let v: TokenType = "PLUS";
let v2: TokenType = "DIVIDE";

assert(v === "PLUS")
assert(v !== "MINUS")

assert(v === v)
assert(v !== v2)

print("done.")