// `await` of an async function that returns nothing runs it, and in order (#440): the await
// left before its `async.await`, so nothing waited for the task and the awaiting function went
// straight on.
let log = "";
async function step(s: string) { log += s; }
async function twice() {
    await step("b");
    await step("c");
}
class C {
    async run() { log += "m"; }
}
async function main() {
    await step("a");
    await twice();
    for (let i = 0; i < 3; i++) {
        await step("" + i);
    }
    const c = new C();
    await c.run();
    const g = async () => { log += "g"; };
    await g();
    print(log);
    assert(log == "abc012mg", "in order");
    print("done.");
}
