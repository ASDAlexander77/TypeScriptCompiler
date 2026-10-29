// A bigint is a 64-bit signed integer: a literal that does not fit is an error, not wrapped.
function main() {
    const a = 12345678901234567890n;
    print(a);
}
