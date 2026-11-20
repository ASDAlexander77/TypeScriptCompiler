// An `await` in a `finally` block is an error at the `await` ("failed to legalize operation 'ts.Try'").
async function v(): Promise<number> { return 1; }
async function a(): Promise<number> {
    let r: number = 0;
    try { r = 1; } finally { r += await v(); }
    return r;
}
