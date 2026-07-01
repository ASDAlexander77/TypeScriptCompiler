async function sleep(d: number)
{
	print("sleep", d);
}

async function* g() {
  yield 1;
  await sleep(100);
  yield* [2, 3];
  yield* (async function*() {
    await sleep(100);
    yield 4;
  })();
}

async function f() {
  for await (const x of g()) {
    print(x);
  }
}

function main() {

    await f();
    (async function () { await f(); })();

    print("done.");
}