import * as fs from 'fs/promises';
import {execSync} from 'child_process';

(async () => {
    try {
        const fld = "G:/Dev/TypeScript/tests/cases/compiler";
        const files = await fs.readdir(fld);
        for await (const file of files) {
            console.log("... read file: " + file);
            //const data = await fs.readFile(fld + "/" + file);

            //console.log("... file data: " + data);
            const output1 = execSync("ts-node scannertree.ts " + fld + "/" + file);
            const output2 = execSync("C:/dev/TypeScriptCompiler/__build/tsc/tsc-new-parser/Debug/tsc-new-parser.exe " + fld + "/" + file);

            const output1_str = output1.toString().split("\n");
            const output2_str = output2.toString().split("\n");

            for (let i = 0; i < output1_str.length; i++)
            {
                const o1 = output1_str[i].trim();
                const o2 = output2_str[i].trim();

                if (o1 != o2)
                {
                    console.log("Output TS:", o1);
                    console.log("Output c++ scanner:", o2);
                    throw "File mismatched " + file;
                }
            }
        }
    }
    catch (err) {
        console.error(err);
    }
})();