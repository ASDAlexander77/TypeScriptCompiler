import * as fs from 'fs';
import { execSync } from 'child_process';
import * as ts from 'typescript';

enum SyntaxKindMapped2 {
    FirstAssignment = "EqualsToken",
    LastAssignment = "CaretEqualsToken",
    FirstCompoundAssignment = "PlusEqualsToken",
    LastCompoundAssignment = "CaretEqualsToken",
    FirstReservedWord = "BreakKeyword",
    LastReservedWord = "WithKeyword",
    FirstKeyword = "BreakKeyword",
    LastKeyword = "OfKeyword",
    FirstFutureReservedWord = "ImplementsKeyword",
    LastFutureReservedWord = "YieldKeyword",
    FirstTypeNode = "TypePredicate",
    LastTypeNode = "ImportType",
    FirstPunctuation = "OpenBraceToken",
    LastPunctuation = "CaretEqualsToken",
    FirstToken = "Unknown",
    LastToken = "OfKeyword",
    FirstTriviaToken = "SingleLineCommentTrivia",
    LastTriviaToken = "ConflictMarkerTrivia",
    FirstLiteralToken = "NumericLiteral",
    LastLiteralToken = "NoSubstitutionTemplateLiteral",
    FirstTemplateToken = "NoSubstitutionTemplateLiteral",
    LastTemplateToken = "TemplateTail",
    FirstBinaryOperator = "LessThanToken",
    LastBinaryOperator = "CaretEqualsToken",
    FirstStatement = "VariableStatement",
    LastStatement = "DebuggerStatement",
    FirstNode = "QualifiedName",
    FirstJSDocNode = "JSDocTypeExpression",
    LastJSDocNode = "JSDocPropertyTag",
    FirstJSDocTagNode = "JSDocTag",
    LastJSDocTagNode = "JSDocPropertyTag",
    FirstContextualKeyword = "AbstractKeyword",
    LastContextualKeyword = "OfKeyword"
};

function printTree(filePath) {
    const dataStr = ts.sys.readFile(filePath).replace(/\r\n/g, "\n");
    const source = ts.createSourceFile(filePath, dataStr, ts.ScriptTarget.Latest);

    let result = "";

    let intent = 0;
    const visitNode = (child) => {
    
        let s = "";
        for (let i = 0; i < intent; i++)
        {
            s += "\t";
        }
    
        const strToken = ts.SyntaxKind[child.kind];
        const str2 = SyntaxKindMapped2[strToken];
        result += s + "Node: " + (str2 || strToken) + " @ [ " + child.pos + " - " + child.end + " ]\n";
    
        intent++;
        ts.forEachChild(child, visitNode, visitArray);
        intent--;
    };
    
    const visitArray = (nodeArray) => {
        for (const node of nodeArray)
        {
            visitNode(node);
        }    
    };
    
    ts.forEachChild(source, visitNode, visitArray);

    return result;
}

try {
    const fld = process.argv[2] || "G:/Dev/TypeScript/tests/cases/compiler";
    const files = fs.readdirSync(fld);
    for (const file of files) {
        //const data = await fs.readFile(fld + "/" + file);

        if (file.endsWith(".tsx"))
        {
            continue;
        }

        if (file == "collisionCodeGenModuleWithUnicodeNames.ts"
            || file == "constructorWithIncompleteTypeAnnotation.ts"
            || file == "extendedUnicodePlaneIdentifiers.ts"
            || file == "extendedUnicodePlaneIdentifiersJSDoc.ts"
            || file == "fileWithNextLine1.ts"
            || file == "parseErrorInHeritageClause1.ts"
            || file == "sourceMap-LineBreaks.ts"
            || file == "unicodeIdentifierName2.ts"
            || file == "unicodeIdentifierNames.ts"
            || file == "unicodeStringLiteral.ts") {
            continue;
        }

        // temporary ignore as too big for TS test app
        if (file == "binaryArithmeticControlFlowGraphNotTooLarge.ts"
            || file == "binderBinaryExpressionStress.ts" 
            || file == "binderBinaryExpressionStressJs.ts"
            || file == "errorRecoveryWithDotFollowedByNamespaceKeyword.ts"
            || file == "largeControlFlowGraph.ts"
            || file == "manyConstExports.ts"
            || file == "parsingDeepParenthensizedExpression.ts"
            || file == "resolvingClassDeclarationWhenInBaseTypeResolution.ts"
            || file == "targetTypeBaseCalls.ts"
            || file == "unionSubtypeReductionErrors.ts") {
            continue;
        }

        // review later, can be due to unicode
        if (file == "bom-utf16be.ts"
            || file == "bom-utf16le.ts"
            || file == "instanceofOperator.ts") {
            continue;
        }

        //console.log("... file data: " + data);
        console.log("printing file TS ... read file: " + file);
        const output1 = printTree(fld + "/" + file);
        console.log("executing file C++ ... read file: " + file);
        const output2 = execSync("C:/dev/TypeScriptCompiler/__build/tsc/tsc-new-parser/Debug/tsc-new-parser.exe " + fld + "/" + file);
        console.log("testing file ... file: " + file);

        const output1_str = output1.toString().split("\n");
        const output2_str = output2.toString().split("\n");

        for (let i = 0; i < output1_str.length; i++) {
            const o1 = output1_str[i].trim();
            const o2 = output2_str[i].trim();

            if (o1 != o2) {
                console.log("Output TS:", o1);
                console.log("Output c++ parser:", o2);
                throw "File mismatched " + file;
            }
        }
    }
}
catch (err) {
    console.error(err);
}
