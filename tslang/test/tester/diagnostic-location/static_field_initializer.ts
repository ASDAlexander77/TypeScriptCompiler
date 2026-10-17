// A static field's failed initializer compiled with no diagnostic: the field was registered
// without a type, and the error, postponed like every message, was never shown.

class Args {
    static count = 1;
    static argv: Ref<string> = null;
}
