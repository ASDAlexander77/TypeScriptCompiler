// An error inside the cast function the compiler generates for a union was reported at the
// generated text's offset taken as a position in this file - inside this comment - rather than
// where the cast is.

function text(x: number | string | null): string {
    const s: string = x;
    return s;
}

print(text("a"));
