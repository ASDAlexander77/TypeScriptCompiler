// Minimal repro for 32-bit async: a non-async main awaits an async function. check-x86-run.sh
// builds this for 32-bit Windows and compares its output exactly: "in g" prints before
// "after await", because main waits on g()'s token before it goes on. Until #497 an await of a
// void async function did not wait, so g() ran on the async runtime after main had resumed (or
// not at all, if the program ended first) and the order was a race.
async function g() { print("in g"); }
function main() { print("start"); await g(); print("after await"); }
