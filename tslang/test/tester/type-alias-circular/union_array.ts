// A type alias that contains itself is an error, not a stack overflow (issue #370).
type Json = number | Json[];

function main() {
    const j: Json = 1;
    print("ok");
}
