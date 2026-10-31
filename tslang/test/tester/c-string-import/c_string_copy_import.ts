// regression test (TSLANG_ISSUES 21): the string a C function returns is copied into a string of the
// program's own (00c_string_copy.ts) also when the .d.ts declaring it was first loaded by a module this
// file imports. The declaration was recorded with that pass's module, and the copy was skipped: both
// stored `strerror` results showed the buffer's last text, and under -mm=rc the count was written into
// C's memory and the release freed it.
/// <reference path="c_string_copy_decls.d.ts" />
import { Other } from "./c_string_copy_lib";

function main() {
    const messages: string[] = [];
    messages.push(strerror(2));
    messages.push(strerror(13));
    assert(messages[0] != messages[1], "each result is a copy of its own");
    assert(messages[0].length > 0 && messages[1].length > 0);
    assert(new Other().name() == "other");
    print(messages[0], "|", messages[1]);
    print("done.");
}
