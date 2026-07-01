type TokenType = "NUMBER" | "PLUS" | "MINUS" | "MULTIPLY" | "DIVIDE" | "LEFT_PAREN" | "RIGHT_PAREN" | "EOF";

class Token {
    type: TokenType;
    value: string;

    constructor(type: TokenType, value: string) {
        this.type = type;
        this.value = value;
    }
}

class Lexer {
    private text: string;
    private pos: number = 0;
    private currentChar: string | null;

    constructor(text: string) {
        this.text = text;
        this.currentChar = this.text.length > 0 ? this.text[0] : null;
    }

    private advance() {
        this.pos++;
        this.currentChar = this.pos < this.text.length ? this.text[this.pos] : null;
    }

    private isDigit(char: string): boolean {
        return char >= "0" && char <= "9";
    }

    private skipWhitespace() {
        while (this.currentChar === " ") {
            this.advance();
        }
    }

    private number(): string {
        let result = "";
        while (this.currentChar !== null && this.isDigit(this.currentChar)) {
            result += this.currentChar;
            this.advance();
        }
        return result;
    }

    getNextToken(): Token {
        while (this.currentChar !== null) {
            if (this.currentChar === " ") {
                this.skipWhitespace();
                continue;
            }

            if (this.isDigit(this.currentChar)) {
                return new Token("NUMBER", this.number());
            }

            if (this.currentChar === "+") {
                this.advance();
                return new Token("PLUS", "+");
            }

            if (this.currentChar === "-") {
                this.advance();
                return new Token("MINUS", "-");
            }

            if (this.currentChar === "*") {
                this.advance();
                return new Token("MULTIPLY", "*");
            }

            if (this.currentChar === "/") {
                this.advance();
                return new Token("DIVIDE", "/");
            }

            if (this.currentChar === "(") {
                this.advance();
                return new Token("LEFT_PAREN", "(");
            }

            if (this.currentChar === ")") {
                this.advance();
                return new Token("RIGHT_PAREN", ")");
            }

            print(`Error: Unknown character: ${this.currentChar}`);
        }

        return new Token("EOF", "");
    }
}

class Parser {
    private lexer: Lexer;
    private currentToken: Token;

    constructor(lexer: Lexer) {
        this.lexer = lexer;
        this.currentToken = this.lexer.getNextToken();
    }

    private eat(tokenType: TokenType) {
        if (this.currentToken.type === tokenType) {
            this.currentToken = this.lexer.getNextToken();
        } else {
            print(`Error: Unexpected token: ${this.currentToken.type}, expected: ${tokenType}`);
        }
    }

    private factor(): number {
        if (this.currentToken.type === "NUMBER") {
            const value = parseInt(this.currentToken.value);
            this.eat("NUMBER");
            return value;
        } else if (this.currentToken.type === "LEFT_PAREN") {
            this.eat("LEFT_PAREN");
            const result = this.expr();
            this.eat("RIGHT_PAREN");
            return result;
        }
        print("Error: Invalid syntax");
    }

    private term(): number {
        let result = this.factor();
        while (this.currentToken.type === "MULTIPLY" || this.currentToken.type === "DIVIDE") {
            if (this.currentToken.type === "MULTIPLY") {
                this.eat("MULTIPLY");
                result *= this.factor();
            } else if (this.currentToken.type === "DIVIDE") {
                this.eat("DIVIDE");
                result /= this.factor();
            }
        }
        return result;
    }

    private expr(): number {
        let result = this.term();
        while (this.currentToken.type === "PLUS" || this.currentToken.type === "MINUS") {
            if (this.currentToken.type === "PLUS") {
                this.eat("PLUS");
                result += this.term();
            } else if (this.currentToken.type === "MINUS") {
                this.eat("MINUS");
                result -= this.term();
            }
        }
        return result;
    }

    parse(): number {
        return this.expr();
    }
}


const input = "(3 + 5) * 2 - 10 / 2";
const lexer = new Lexer(input);
const parser = new Parser(lexer);

const r = parser.parse();
print(`Result: ${r}`);

assert(r === 11, "Test failed");

print("done.");