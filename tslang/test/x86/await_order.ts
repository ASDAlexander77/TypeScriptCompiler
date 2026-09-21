// Minimal repro for 32-bit async: a non-async main awaits an async function. Confirmed at x64
// (docs/superpowers/plans/2026-09-22-32-bit-phase-4b-async.md): "after await" prints before
// "in g" because g() keeps running on the async runtime after main resumes past the await.
// check-x86-run.sh builds this for 32-bit Windows and compares its output exactly.
async function g() { print("in g"); }
function main() { print("start"); await g(); print("after await"); }
