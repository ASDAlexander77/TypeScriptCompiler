// A namespace is no run-time value, in a field of an object literal either.
namespace M { export const a = 1; }
var o = { m: M };
