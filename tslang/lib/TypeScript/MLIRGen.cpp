#include "MLIRGenImpl.h"


// TODO: optimize of amount of calls to detect return types and if it is was calculated before then do not run it all
// the time
static CompileOptions *compileOptionsPtr = nullptr;
CompileOptions& getCompileOptions()
{
    return *compileOptionsPtr;
}

void setCompileOptions(CompileOptions &compileOptions)
{
    compileOptionsPtr = &compileOptions;
}

SourceMgrDiagnosticHandlerEx::SourceMgrDiagnosticHandlerEx(llvm::SourceMgr &mgr, mlir::MLIRContext *ctx) : mlir::SourceMgrDiagnosticHandler(mgr, ctx)
{
}

void SourceMgrDiagnosticHandlerEx::emit(mlir::Diagnostic &diag)
{
    emitDiagnostic(diag);
}

namespace typescript
{
::std::string dumpFromSource(const llvm::StringRef &fileName, const llvm::StringRef &source)
{
    auto showLineCharPos = false;

    Parser parser;
    auto sourceFile = parser.parseSourceFile(stows(static_cast<std::string>(fileName)),
                                             stows(static_cast<std::string>(source)), ScriptTarget::Latest);

    stringstream s;

    FuncT<> visitNode;
    ArrayFuncT<> visitArray;

    auto intent = 0;

    visitNode = [&](Node child) -> Node {
        for (auto i = 0; i < intent; i++)
        {
            s << "\t";
        }

        if (showLineCharPos)
        {
            auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, child->pos);
            auto endLineChar = parser.getLineAndCharacterOfPosition(sourceFile, child->_end);

            s << S("Node: ") << parser.syntaxKindString(child).c_str() << S(" @ [ ") << child->pos << S("(")
              << posLineChar.line + 1 << S(":") << posLineChar.character + 1 << S(") - ") << child->_end << S("(")
              << endLineChar.line + 1 << S(":") << endLineChar.character << S(") ]") << std::endl;
        }
        else
        {
            s << S("Node: ") << parser.syntaxKindString(child).c_str() << S(" @ [ ") << child->pos << S(" - ")
              << child->_end << S(" ]") << std::endl;
        }

        intent++;
        ts::forEachChild(child, visitNode, visitArray);
        intent--;

        return undefined;
    };

    visitArray = [&](NodeArray<Node> array) -> Node {
        for (auto node : array)
        {
            visitNode(node);
        }

        return undefined;
    };

    auto result = forEachChild(sourceFile.as<Node>(), visitNode, visitArray);
    return convertWideToUTF8(s.str());
}

mlir::OwningOpRef<mlir::ModuleOp> mlirGenFromMainSource(const mlir::MLIRContext &context, const llvm::StringRef &fileName,
                                        const llvm::SourceMgr &sourceMgr, CompileOptions &compileOptions)
{
    auto path = llvm::sys::path::parent_path(fileName);
    mlirgen::MLIRGenImpl mlirGenImpl(context, fileName, path, sourceMgr, compileOptions);
    auto [sourceFile, includeFiles] = mlirGenImpl.loadMainSourceFile();
    return mlirGenImpl.mlirGenSourceFile(sourceFile, includeFiles);
}

mlir::OwningOpRef<mlir::ModuleOp> mlirGenFromSource(const mlir::MLIRContext &context, SMLoc &smLoc, const llvm::StringRef &fileName,
                                        const llvm::SourceMgr &sourceMgr, CompileOptions &compileOptions)
{
    auto path = llvm::sys::path::parent_path(fileName);
    mlirgen::MLIRGenImpl mlirGenImpl(context, fileName, path, sourceMgr, compileOptions);
    auto [sourceFile, includeFiles] = mlirGenImpl.loadSourceFile(smLoc);
    return mlirGenImpl.mlirGenSourceFile(sourceFile, includeFiles);
}

} // namespace typescript
