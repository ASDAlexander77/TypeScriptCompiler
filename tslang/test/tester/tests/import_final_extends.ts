// A @final class from another module cannot be extended: its declaration (__decls) keeps @final.
import './export_final_class'

class X extends Counter {}

function main() {
    print(new X().inc());
}
