class Lexer {
    private currentChar: string | null;

    constructor(text: string) {
        this.currentChar = text.length > 0 ? text[0] : null;
    }

    getNextToken() {
        const currentChar = this.currentChar;
        while (currentChar !== null) {
            if (currentChar === " ") {
                continue;
            }

            break;
        }

        return true;
    }
}

print("done.")