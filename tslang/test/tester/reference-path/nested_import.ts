// The program imports nested_middle, which imports nested_bottom. Under test-runner -shared the
// library is named after nested_bottom and holds nested_middle too, which the program only learns
// inside nested_middle: it was declared both from its source and from the library ("redefinition
// of symbol named 'nested_fn'"). The library is imported in its place.
import './nested_middle'

assert(nested_fn() == 3);

print("done.");
