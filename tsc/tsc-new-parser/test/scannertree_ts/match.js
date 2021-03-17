import * as fs from 'fs/promises';
import {execSync} from 'child_process';

(async () => {
    try {
        const fld = "G:/Dev/TypeScript/tests/cases/compiler/";
        const files = await fs.readdir(fld);
        for await (const file of files) {
            console.log("... read file: " + file);
            const data = await fs.readFile(fld + "/" + file);

            //console.log("... file data: " + data);
            const output1 = execSync("ts-node scannertree.ts " + fld + "/" + file);
            const output2 = execSync("C:/dev/TypeScriptCompiler/__build/tsc/tsc-new-parser/Debug/tsc-new-parser.exe " + fld + "/" + file);

            if (output1 != output2)
            {
                console.log("Output TS:", output1);
                console.log("Output c++ scanner:", output2);
                throw "File mismatched " + file;
            }
        }
    }
    catch (err) {
        console.error(err);
    }
})();