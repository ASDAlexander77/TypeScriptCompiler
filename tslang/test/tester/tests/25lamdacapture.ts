let glb1 = 0;

function testInnerLambdaCapture() {
    print("testInnerLambdaCapture");
    glb1 = 0;
    let a = 7;
    let g = () => {
        let h = () => {
            glb1 += a;
        };
        h();
    };
    g();
    assert(glb1 == 7, "7");
}

function main() {
    testInnerLambdaCapture();
    print("done.");
}
