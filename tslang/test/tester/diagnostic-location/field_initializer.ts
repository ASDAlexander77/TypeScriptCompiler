// An error in a field's initializer was reported at the start of the file: the assignment the
// compiler generates for it in the constructor had no position.

class Args {
    count = 1;
    protected argv: Ref<string> = null;
}
