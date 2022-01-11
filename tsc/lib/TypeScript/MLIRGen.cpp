#define ENABLE_RTTI true
#define ALL_METHODS_VIRTUAL true
#define USE_BOUND_FUNCTION_FOR_OBJECTS true
#ifdef GC_ENABLE
#define ADD_GC_ATTRIBUTE true
#endif
#define MODULE_AS_NAMESPACE true

#define DEBUG_TYPE "mlir"

#include "TypeScript/MLIRGen.h"
#include "TypeScript/Config.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/MLIRLogic/MLIRCodeLogic.h"
#include "TypeScript/MLIRLogic/MLIRGenContext.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"
#ifdef WIN_EXCEPTION
#include "TypeScript/MLIRLogic/MLIRRTTIHelperVCWin32.h"
#else
#include "TypeScript/MLIRLogic/MLIRRTTIHelperVCLinux.h"
#endif
#include "TypeScript/VisitorAST.h"

#include "TypeScript/DOM.h"
#include "TypeScript/Defines.h"

// parser includes
#include "file_helper.h"
#include "node_factory.h"
#include "parser.h"
#include "utilities.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Diagnostics.h"
#ifdef ENABLE_ASYNC
#include "mlir/Dialect/Async/IR/Async.h"
#endif

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <numeric>

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::makeArrayRef;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

// TODO: optimize of amount of calls to detect return types and if it is was calculated before then do not run it all the time

namespace
{

/// Implementation of a simple MLIR emission from the TypeScript AST.
///
/// This will emit operations that are specific to the TypeScript language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl
{
  public:
    MLIRGenImpl(const mlir::MLIRContext &context, CompileOptions compileOptions)
        : hasErrorMessages(false), builder(&const_cast<mlir::MLIRContext &>(context)), compileOptions(compileOptions)
    {
        fileName = "<unknown>";
        rootNamespace = currentNamespace = std::make_shared<NamespaceInfo>();
    }

    MLIRGenImpl(const mlir::MLIRContext &context, const llvm::StringRef &fileNameParam, CompileOptions compileOptions)
        : hasErrorMessages(false), builder(&const_cast<mlir::MLIRContext &>(context)), compileOptions(compileOptions)
    {
        fileName = fileNameParam;
        rootNamespace = currentNamespace = std::make_shared<NamespaceInfo>();
    }

    mlir::ModuleOp mlirGenSourceFile(SourceFile module)
    {
        if (failed(mlirGenCodeGenInit(module)))
        {
            return nullptr;
        }

        SymbolTableScopeT varScope(symbolTable);
        llvm::ScopedHashTableScope<StringRef, NamespaceInfo::TypePtr> fullNamespacesMapScope(fullNamespacesMap);
        llvm::ScopedHashTableScope<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMapScope(fullNameGlobalsMap);
        llvm::ScopedHashTableScope<StringRef, ClassInfo::TypePtr> fullNameClassesMapScope(fullNameClassesMap);
        llvm::ScopedHashTableScope<StringRef, GenericClassInfo::TypePtr> fullNameGenericClassesMapScope(fullNameGenericClassesMap);
        llvm::ScopedHashTableScope<StringRef, InterfaceInfo::TypePtr> fullNameInterfacesMapScope(fullNameInterfacesMap);
        llvm::ScopedHashTableScope<StringRef, GenericInterfaceInfo::TypePtr> fullNameGenericInterfacesMapScope(
            fullNameGenericInterfacesMap);

        if (mlir::succeeded(mlirDiscoverAllDependencies(module)) && mlir::succeeded(mlirCodeGenModuleWithDiagnostics(module)))
        {
            return theModule;
        }

        return nullptr;
    }

  private:
    mlir::LogicalResult mlirGenCodeGenInit(SourceFile module)
    {
        sourceFile = module;

        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        theModule = mlir::ModuleOp::create(loc(module), fileName);
        builder.setInsertionPointToStart(theModule.getBody());

        return mlir::success();
    }

    mlir::LogicalResult mlirDiscoverAllDependencies(SourceFile module)
    {
        mlir::SmallVector<mlir::Diagnostic *> postponedMessages;
        mlir::ScopedDiagnosticHandler diagHandler(builder.getContext(), [&](mlir::Diagnostic &diag) {
            // suppress all
            if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
            {
                hasErrorMessages = true;
            }

            postponedMessages.push_back(new mlir::Diagnostic(std::move(diag)));
        });

        llvm::ScopedHashTableScope<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMapScope(fullNameGlobalsMap);

        // Process of discovery here
        GenContext genContextPartial{};
        genContextPartial.allowPartialResolve = true;
        genContextPartial.dummyRun = true;
        genContextPartial.cleanUps = new mlir::SmallVector<mlir::Block *>();
        genContextPartial.unresolved = new mlir::SmallVector<std::pair<mlir::Location, std::string>>();
        auto notResolved = 0;
        do
        {
            // clear previous errors
            hasErrorMessages = false;
            for (auto diag : postponedMessages)
            {
                delete diag;
            }

            postponedMessages.clear();
            genContextPartial.unresolved->clear();

            // main cycles
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;
            for (auto &statement : module->statements)
            {
                if (statement->processed)
                {
                    continue;
                }

                if (failed(mlirGen(statement, genContextPartial)))
                {
                    notResolved++;
                }
                else
                {
                    statement->processed = true;
                }
            }

            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                if (genContextPartial.unresolved->size() == 0)
                {
                    theModule.emitError("can't resolve dependencies");
                }

                for (auto unresolvedRef : *genContextPartial.unresolved)
                {
                    emitError(std::get<0>(unresolvedRef), "can't resolve reference: ") << std::get<1>(unresolvedRef);
                }

                break;
            }

        } while (notResolved > 0);

        genContextPartial.clean();
        genContextPartial.cleanUnresolved();

        // clean up
        theModule.getBody()->clear();

        // clear state
        for (auto &statement : module->statements)
        {
            statement->processed = false;
        }

        if (hasErrorMessages)
        {
            // print errors
            for (auto diag : postponedMessages)
            {
                // we show messages when they metter
                if (notResolved)
                {
                    publishDiagnostic(*diag);
                }

                delete diag;
            }

            postponedMessages.clear();

            // we return error when we can't generate code
            if (notResolved)
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirCodeGenModule(SourceFile module)
    {
        hasErrorMessages = false;

        llvm::ScopedHashTableScope<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMapScope(fullNameGlobalsMap);

        // Process generating here
        GenContext genContext{};
        for (auto &statement : module->statements)
        {
            if (failed(mlirGen(statement, genContext)))
            {
                return mlir::failure();
            }
        }

        if (hasErrorMessages)
        {
            return mlir::failure();
        }

        // Verify the module after we have finished constructing it, this will check
        // the structural properties of the IR and invoke any specific verifiers we
        // have on the TypeScript operations.
        if (failed(mlir::verify(theModule)))
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! broken module: \n" << theModule << "\n";);

            theModule.emitError("module verification error");
            return mlir::failure();
        }

        return mlir::success();
    }

    void publishDiagnostic(mlir::Diagnostic &diag)
    {
        auto printMsg = [](llvm::raw_fd_ostream &os, mlir::Diagnostic &diag, const char *msg) {
            if (!diag.getLocation().isa<mlir::UnknownLoc>())
                os << diag.getLocation() << ": ";
            os << msg;

            // The default behavior for errors is to emit them to stderr.
            os << diag << '\n';
            os.flush();
        };

        switch (diag.getSeverity())
        {
        case mlir::DiagnosticSeverity::Note:
            printMsg(llvm::outs(), diag, "note: ");
            for (auto &note : diag.getNotes())
            {
                printMsg(llvm::outs(), note, "note: ");
            }

            break;
        case mlir::DiagnosticSeverity::Warning:
            printMsg(llvm::outs(), diag, "warning: ");
            break;
        case mlir::DiagnosticSeverity::Error:
            hasErrorMessages = true;
            printMsg(llvm::errs(), diag, "error: ");
            break;
        case mlir::DiagnosticSeverity::Remark:
            printMsg(llvm::outs(), diag, "information: ");
            break;
        }
    }

    mlir::LogicalResult mlirCodeGenModuleWithDiagnostics(SourceFile module)
    {
        mlir::ScopedDiagnosticHandler diagHandler(builder.getContext(), [&](mlir::Diagnostic &diag) { publishDiagnostic(diag); });

        if (failed(mlirCodeGenModule(module)) || hasErrorMessages)
        {
            return mlir::failure();
        }

        return mlir::success();
    }

    mlir::LogicalResult registerNamespace(llvm::StringRef namePtr, bool isFunctionNamespace = false)
    {
        auto fullNamePtr = getFullNamespaceName(namePtr);
        auto &namespacesMap = getNamespaceMap();
        auto it = namespacesMap.find(namePtr);
        if (it == namespacesMap.end())
        {
            auto newNamespacePtr = std::make_shared<NamespaceInfo>();
            newNamespacePtr->name = namePtr;
            newNamespacePtr->fullName = fullNamePtr;
            newNamespacePtr->namespaceType = getNamespaceType(fullNamePtr);
            newNamespacePtr->parentNamespace = currentNamespace;
            newNamespacePtr->isFunctionNamespace = isFunctionNamespace;
            namespacesMap.insert({namePtr, newNamespacePtr});
            if (!isFunctionNamespace && !fullNamespacesMap.count(fullNamePtr))
            {
                // TODO: full investigation needed, if i register function namespace as full namespace, it will fail running
                fullNamespacesMap.insert(fullNamePtr, newNamespacePtr);
            }

            currentNamespace = newNamespacePtr;
        }
        else
        {
            currentNamespace = it->getValue();
        }

        return mlir::success();
    }

    mlir::LogicalResult exitNamespace()
    {
        // TODO: it will increase reference count, investigate how to fix it
        currentNamespace = currentNamespace->parentNamespace;
        return mlir::success();
    }

    mlir::LogicalResult mlirGenNamespace(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext)
    {
        auto location = loc(moduleDeclarationAST);

        auto namespaceName = MLIRHelper::getName(moduleDeclarationAST->name, stringAllocator);
        auto namePtr = namespaceName;

        registerNamespace(namePtr);

        auto result = mlirGenBody(moduleDeclarationAST->body, genContext);

        exitNamespace();

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext)
    {
#ifdef MODULE_AS_NAMESPACE
        return mlirGenNamespace(moduleDeclarationAST, genContext);
#else
        auto isNamespace = (moduleDeclarationAST->flags & NodeFlags::Namespace) == NodeFlags::Namespace;
        auto isNestedNamespace = (moduleDeclarationAST->flags & NodeFlags::NestedNamespace) == NodeFlags::NestedNamespace;
        if (isNamespace || isNestedNamespace)
        {
            return mlirGenNamespace(moduleDeclarationAST, genContext);
        }

        auto location = loc(moduleDeclarationAST);

        auto moduleName = MLIRHelper::getName(moduleDeclarationAST->name);

        auto moduleOp = builder.create<mlir::ModuleOp>(location, StringRef(moduleName));

        builder.setInsertionPointToStart(&moduleOp.body().front());

        // save module theModule
        auto parentModule = theModule;
        theModule = moduleOp;

        GenContext moduleGenContext{};
        auto result = mlirGenBody(moduleDeclarationAST->body, genContext);

        // restore
        theModule = parentModule;

        builder.setInsertionPointAfter(moduleOp);

        return result;
#endif
    }

    mlir::LogicalResult mlirGenBody(Node body, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)body;
        if (kind == SyntaxKind::Block)
        {
            return mlirGen(body.as<Block>(), genContext);
        }

        if (kind == SyntaxKind::ModuleBlock)
        {
            return mlirGen(body.as<ModuleBlock>(), genContext);
        }

        if (body.is<Statement>())
        {
            return mlirGen(body.as<Statement>(), genContext);
        }

        if (body.is<Expression>())
        {
            auto result = mlirGen(body.as<Expression>(), genContext);
            if (result)
            {
                return mlirGenReturnValue(loc(body), result, false, genContext);
            }

            builder.create<mlir_ts::ReturnOp>(loc(body));
            return mlir::success();
        }

        llvm_unreachable("unknown body type");
    }

    mlir::LogicalResult mlirGen(ModuleBlock moduleBlockAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        // clear up state
        for (auto &statement : moduleBlockAST->statements)
        {
            statement->processed = false;
        }

        auto notResolved = 0;
        do
        {
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;
            for (auto &statement : moduleBlockAST->statements)
            {
                if (statement->processed)
                {
                    continue;
                }

                if (failed(mlirGen(statement, genContext)))
                {
                    notResolved++;
                }
                else
                {
                    statement->processed = true;
                }
            }

            // repeat if not all resolved
            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                // class can depends on other class declarations
                theModule.emitError("can't resolve dependencies in namespace");
                return mlir::failure();
            }
        } while (notResolved > 0);

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(Block blockAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        if (genContext.generatedStatements.size() > 0)
        {
            // auto generated code
            for (auto &statement : genContext.generatedStatements)
            {
                if (failed(mlirGen(statement, genContext)))
                {
                    return mlir::failure();
                }
            }

            // clean up
            const_cast<GenContext &>(genContext).generatedStatements.clear();
        }

        for (auto &statement : blockAST->statements)
        {
            if (genContext.skipProcessed && statement->processed)
            {
                continue;
            }

            if (failed(mlirGen(statement, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(Statement statementAST, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)statementAST;
        if (kind == SyntaxKind::FunctionDeclaration)
        {
            return mlirGen(statementAST.as<FunctionDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ExpressionStatement)
        {
            return mlirGen(statementAST.as<ExpressionStatement>(), genContext);
        }
        else if (kind == SyntaxKind::VariableStatement)
        {
            return mlirGen(statementAST.as<VariableStatement>(), genContext);
        }
        else if (kind == SyntaxKind::IfStatement)
        {
            return mlirGen(statementAST.as<IfStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ReturnStatement)
        {
            return mlirGen(statementAST.as<ReturnStatement>(), genContext);
        }
        else if (kind == SyntaxKind::LabeledStatement)
        {
            return mlirGen(statementAST.as<LabeledStatement>(), genContext);
        }
        else if (kind == SyntaxKind::DoStatement)
        {
            return mlirGen(statementAST.as<DoStatement>(), genContext);
        }
        else if (kind == SyntaxKind::WhileStatement)
        {
            return mlirGen(statementAST.as<WhileStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ForStatement)
        {
            return mlirGen(statementAST.as<ForStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ForInStatement)
        {
            return mlirGen(statementAST.as<ForInStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ForOfStatement)
        {
            return mlirGen(statementAST.as<ForOfStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ContinueStatement)
        {
            return mlirGen(statementAST.as<ContinueStatement>(), genContext);
        }
        else if (kind == SyntaxKind::BreakStatement)
        {
            return mlirGen(statementAST.as<BreakStatement>(), genContext);
        }
        else if (kind == SyntaxKind::SwitchStatement)
        {
            return mlirGen(statementAST.as<SwitchStatement>(), genContext);
        }
        else if (kind == SyntaxKind::ThrowStatement)
        {
            return mlirGen(statementAST.as<ThrowStatement>(), genContext);
        }
        else if (kind == SyntaxKind::TryStatement)
        {
            return mlirGen(statementAST.as<TryStatement>(), genContext);
        }
        else if (kind == SyntaxKind::LabeledStatement)
        {
            return mlirGen(statementAST.as<LabeledStatement>(), genContext);
        }
        else if (kind == SyntaxKind::TypeAliasDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<TypeAliasDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::Block)
        {
            return mlirGen(statementAST.as<Block>(), genContext);
        }
        else if (kind == SyntaxKind::EnumDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<EnumDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ClassDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<ClassLikeDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::InterfaceDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<InterfaceDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ImportEqualsDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<ImportEqualsDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ModuleDeclaration)
        {
            return mlirGen(statementAST.as<ModuleDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::DebuggerStatement)
        {
            return mlirGen(statementAST.as<DebuggerStatement>(), genContext);
        }
        else if (kind == SyntaxKind::EmptyStatement ||
                 kind == SyntaxKind::Unknown /*TODO: temp solution to treat null statements as empty*/)
        {
            return mlir::success();
        }

        llvm_unreachable("unknown statement type");
    }

    mlir::LogicalResult mlirGen(ExpressionStatement expressionStatementAST, const GenContext &genContext)
    {
        mlirGen(expressionStatementAST->expression, genContext);
        return mlir::success();
    }

    mlir::Value mlirGen(Expression expressionAST, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)expressionAST;
        if (kind == SyntaxKind::NumericLiteral)
        {
            return mlirGen(expressionAST.as<NumericLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::StringLiteral)
        {
            return mlirGen(expressionAST.as<ts::StringLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::NoSubstitutionTemplateLiteral)
        {
            return mlirGen(expressionAST.as<NoSubstitutionTemplateLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::BigIntLiteral)
        {
            return mlirGen(expressionAST.as<BigIntLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::NullKeyword)
        {
            return mlirGen(expressionAST.as<NullLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::TrueKeyword)
        {
            return mlirGen(expressionAST.as<TrueLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::FalseKeyword)
        {
            return mlirGen(expressionAST.as<FalseLiteral>(), genContext);
        }
        else if (kind == SyntaxKind::ArrayLiteralExpression)
        {
            return mlirGen(expressionAST.as<ArrayLiteralExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ObjectLiteralExpression)
        {
            return mlirGen(expressionAST.as<ObjectLiteralExpression>(), genContext);
        }
        else if (kind == SyntaxKind::Identifier)
        {
            return mlirGen(expressionAST.as<Identifier>(), genContext);
        }
        else if (kind == SyntaxKind::CallExpression)
        {
            return mlirGen(expressionAST.as<CallExpression>(), genContext);
        }
        else if (kind == SyntaxKind::SpreadElement)
        {
            return mlirGen(expressionAST.as<SpreadElement>(), genContext);
        }
        else if (kind == SyntaxKind::BinaryExpression)
        {
            return mlirGen(expressionAST.as<BinaryExpression>(), genContext);
        }
        else if (kind == SyntaxKind::PrefixUnaryExpression)
        {
            return mlirGen(expressionAST.as<PrefixUnaryExpression>(), genContext);
        }
        else if (kind == SyntaxKind::PostfixUnaryExpression)
        {
            return mlirGen(expressionAST.as<PostfixUnaryExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ParenthesizedExpression)
        {
            return mlirGen(expressionAST.as<ParenthesizedExpression>(), genContext);
        }
        else if (kind == SyntaxKind::TypeOfExpression)
        {
            return mlirGen(expressionAST.as<TypeOfExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ConditionalExpression)
        {
            return mlirGen(expressionAST.as<ConditionalExpression>(), genContext);
        }
        else if (kind == SyntaxKind::PropertyAccessExpression)
        {
            return mlirGen(expressionAST.as<PropertyAccessExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ElementAccessExpression)
        {
            return mlirGen(expressionAST.as<ElementAccessExpression>(), genContext);
        }
        else if (kind == SyntaxKind::FunctionExpression)
        {
            return mlirGen(expressionAST.as<FunctionExpression>(), genContext);
        }
        else if (kind == SyntaxKind::ArrowFunction)
        {
            return mlirGen(expressionAST.as<ArrowFunction>(), genContext);
        }
        else if (kind == SyntaxKind::TypeAssertionExpression)
        {
            return mlirGen(expressionAST.as<TypeAssertion>(), genContext);
        }
        else if (kind == SyntaxKind::AsExpression)
        {
            return mlirGen(expressionAST.as<AsExpression>(), genContext);
        }
        else if (kind == SyntaxKind::TemplateExpression)
        {
            return mlirGen(expressionAST.as<TemplateLiteralLikeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TaggedTemplateExpression)
        {
            return mlirGen(expressionAST.as<TaggedTemplateExpression>(), genContext);
        }
        else if (kind == SyntaxKind::NewExpression)
        {
            return mlirGen(expressionAST.as<NewExpression>(), genContext);
        }
        else if (kind == SyntaxKind::DeleteExpression)
        {
            mlirGen(expressionAST.as<DeleteExpression>(), genContext);
            return mlir::Value();
        }
        else if (kind == SyntaxKind::ThisKeyword)
        {
            return mlirGen(loc(expressionAST), THIS_NAME, genContext);
        }
        else if (kind == SyntaxKind::SuperKeyword)
        {
            return mlirGen(loc(expressionAST), SUPER_NAME, genContext);
        }
        else if (kind == SyntaxKind::VoidExpression)
        {
            return mlirGen(expressionAST.as<VoidExpression>(), genContext);
        }
        else if (kind == SyntaxKind::YieldExpression)
        {
            return mlirGen(expressionAST.as<YieldExpression>(), genContext);
        }
        else if (kind == SyntaxKind::AwaitExpression)
        {
            return mlirGen(expressionAST.as<AwaitExpression>(), genContext);
        }
        else if (kind == SyntaxKind::NonNullExpression)
        {
            return mlirGen(expressionAST.as<NonNullExpression>(), genContext);
        }
        else if (kind == SyntaxKind::Unknown /*TODO: temp solution to treat null expr as empty expr*/)
        {
            return mlir::Value();
        }
        else if (kind == SyntaxKind::OmittedExpression /*TODO: temp solution to treat null expr as empty expr*/)
        {
            return mlir::Value();
        }

        llvm_unreachable("unknown expression");
    }

    mlir_ts::FuncOp instantiateSpecializedFunctionType(mlir::Location location, StringRef name, NodeArray<TypeNode> typeArguments,
                                                       const GenContext &genContext)
    {
        if (getGenericFunctionMap().count(name))
        {
            auto functionGenericTypeInfo = getGenericFunctionMap().lookup(name);

            GenContext genericTypeGenContext(genContext);
            auto typeParams = functionGenericTypeInfo->typeParams;
            if (typeArguments && typeArguments.size() > 0)
            {
                // create typeParamsWithArgs from typeArguments
                if (mlir::failed(zipTypeParametersWithArguments(location, typeParams, typeArguments,
                                                                genericTypeGenContext.typeParamsWithArgs, genContext)))
                {
                    return mlir_ts::FuncOp();
                }
            }
            else if (genericTypeGenContext.callOperands.size() > 0)
            {
                GenContext funcGenContext(genContext);
                // we need to map generic parameters to generic types to be able to resolve function
                for (auto typeParam : typeParams)
                {
                    funcGenContext.typeAliasMap.insert({typeParam->getName(), getNamedGenericType(typeParam->getName())});
                }

                auto funcOpWithFuncProto = mlirGenFunctionPrototype(functionGenericTypeInfo->functionDeclaration, funcGenContext);
                auto &funcOp = std::get<0>(funcOpWithFuncProto);
                auto &funcProto = std::get<1>(funcOpWithFuncProto);
                auto result = std::get<2>(funcOpWithFuncProto);
                if (!result || !funcOp)
                {
                    return mlir_ts::FuncOp();
                }

                // TODO: we have func params.
                auto index = 0;
                for (auto argInfo : funcProto->getArgs())
                {
                    auto type = argInfo->getType();
                    auto typeParam = typeParams[index];
                    auto op = genContext.callOperands[index++];

                    if (auto namedGenericType = type.dyn_cast<mlir_ts::NamedGenericType>())
                    {
                        if (namedGenericType.getName().getValue() != typeParam->getName())
                        {
                            emitError(location) << "generic parameter names mismatch.";
                            return mlir_ts::FuncOp();
                        }
                    }
                    else
                    {
                        llvm_unreachable("not implemented yet");
                    }

                    // TODO: if type in generic is complex such Array<T>. finish logic to unwrap T and assing it to unwrapped operand type

                    if (mlir::failed(
                            zipTypeParameterWithArgument(location, genericTypeGenContext.typeParamsWithArgs, typeParam, op.getType())))
                    {
                        return mlir_ts::FuncOp();
                    }
                }
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            // create new instance of interface with TypeArguments
            auto res = mlirGenFunctionLikeDeclaration(functionGenericTypeInfo->functionDeclaration, genericTypeGenContext);
            if (mlir::failed(std::get<0>(res)))
            {
                return mlir_ts::FuncOp();
            }

            return std::get<1>(res);
        }

        return mlir_ts::FuncOp();
    }

    mlir::Type instantiateSpecializedClassType(mlir::Location location, mlir_ts::ClassType genericClassType,
                                               NodeArray<TypeNode> typeArguments, const GenContext &genContext)
    {
        auto fullNameGenericClassTypeName = genericClassType.getName().getValue();
        if (hasGenericClassInfoByFullName(fullNameGenericClassTypeName))
        {
            auto genericClassInfo = getGenericClassInfoByFullName(fullNameGenericClassTypeName);

            GenContext genericTypeGenContext(genContext);
            auto typeParams = genericClassInfo->typeParams;
            if (mlir::failed(zipTypeParametersWithArguments(location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs,
                                                            genContext)))
            {
                return mlir::Type();
            }

            // create new instance of interface with TypeArguments
            if (mlir::failed(mlirGen(genericClassInfo->classDeclaration, genericTypeGenContext)))
            {
                return mlir::Type();
            }

            // get instance of generic interface type
            auto specType = getSpecializationClassType(genericClassInfo, genericTypeGenContext);
            return specType;
        }
        else
        {
            // can't find generic instance
            return mlir::Type();
        }
    }

    mlir::Type instantiateSpecializedInterfaceType(mlir::Location location, mlir_ts::InterfaceType genericInterfaceType,
                                                   NodeArray<TypeNode> typeArguments, const GenContext &genContext)
    {
        auto fullNameGenericInterfaceTypeName = genericInterfaceType.getName().getValue();
        if (hasGenericInterfaceInfoByFullName(fullNameGenericInterfaceTypeName))
        {
            auto genericInterfaceInfo = getGenericInterfaceInfoByFullName(fullNameGenericInterfaceTypeName);

            GenContext genericTypeGenContext(genContext);
            auto typeParams = genericInterfaceInfo->typeParams;
            if (mlir::failed(zipTypeParametersWithArguments(location, typeParams, typeArguments, genericTypeGenContext.typeParamsWithArgs,
                                                            genContext)))
            {
                return mlir::Type();
            }

            // create new instance of interface with TypeArguments
            if (mlir::failed(mlirGen(genericInterfaceInfo->interfaceDeclaration, genericTypeGenContext)))
            {
                return mlir::Type();
            }

            // get instance of generic interface type
            auto specType = getSpecializationInterfaceType(genericInterfaceInfo, genericTypeGenContext);
            return specType;
        }
        else
        {
            // can't find generic instance
            return mlir::Type();
        }
    }

    mlir::Value mlirGenSpecialized(mlir::Location location, mlir::Value genResult, NodeArray<TypeNode> typeArguments,
                                   const GenContext &genContext)
    {
        // TODO: process generic function
        if (auto symbolOp = genResult.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            if (!symbolOp.getType().isa<mlir_ts::GenericType>())
            {
                // it is not generic function reference
                return genResult;
            }

            // create new function instance
            auto funcOp = instantiateSpecializedFunctionType(location, symbolOp.identifierAttr().getValue(), typeArguments, genContext);
            if (!funcOp)
            {
                return mlir::Value();
            }

            auto symbOp = builder.create<mlir_ts::SymbolRefOp>(location, funcOp.getType(),
                                                               mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
            return symbOp;
        }

        if (auto classOp = genResult.getDefiningOp<mlir_ts::ClassRefOp>())
        {
            auto classType = classOp.getType();
            auto specType = instantiateSpecializedClassType(location, classType, typeArguments, genContext);
            if (auto specClassType = specType.dyn_cast_or_null<mlir_ts::ClassType>())
            {
                return builder.create<mlir_ts::ClassRefOp>(
                    location, specClassType, mlir::FlatSymbolRefAttr::get(builder.getContext(), specClassType.getName().getValue()));
            }

            // can't find generic instance
            return mlir::Value();
        }

        if (auto ifaceOp = genResult.getDefiningOp<mlir_ts::InterfaceRefOp>())
        {
            auto interfaceType = ifaceOp.getType();
            auto specType = instantiateSpecializedInterfaceType(location, interfaceType, typeArguments, genContext);
            if (auto specInterfaceType = specType.dyn_cast<mlir_ts::InterfaceType>())
            {
                return builder.create<mlir_ts::InterfaceRefOp>(
                    location, specInterfaceType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), specInterfaceType.getName().getValue()));
            }

            // can't find generic instance
            return mlir::Value();
        }

        return mlir::Value();
    }

    mlir::Value mlirGen(Expression expression, NodeArray<TypeNode> typeArguments, const GenContext &genContext)
    {
        auto genResult = mlirGen(expression, genContext);
        if (typeArguments.size() == 0)
        {
            return genResult;
        }

        auto location = loc(expression);

        auto specializedResult = mlirGenSpecialized(location, genResult, typeArguments, genContext);
        if (specializedResult)
        {
            return specializedResult;
        }

        return genResult;
    }

    mlir::Value mlirGen(ExpressionWithTypeArguments expressionWithTypeArgumentsAST, const GenContext &genContext)
    {
        return mlirGen(expressionWithTypeArgumentsAST->expression, expressionWithTypeArgumentsAST->typeArguments, genContext);
    }

    mlir::Value registerVariableInThisContext(mlir::Location location, StringRef name, mlir::Type type, const GenContext &genContext)
    {
        if (genContext.passResult)
        {
            MLIRTypeHelper mth(builder.getContext());
            // create new type with added field
            genContext.passResult->extraFieldsInThisContext.push_back({mth.TupleFieldName(name), type});
            return mlir::Value();
        }

        // resolve object property

        NodeFactory nf(NodeFactoryFlags::None);
        // load this.<var name>
        auto _this = nf.createToken(SyntaxKind::ThisKeyword);
        auto _name = nf.createIdentifier(stows(std::string(name)));
        auto _this_name = nf.createPropertyAccessExpression(_this, _name);

        auto thisVarValue = mlirGen(_this_name, genContext);

        assert(thisVarValue);

        MLIRCodeLogic mcl(builder);
        auto thisVarValueRef = mcl.GetReferenceOfLoadOp(thisVarValue);

        assert(thisVarValueRef);

        return thisVarValueRef;
    }

    bool isConstValue(mlir::Value init)
    {
        if (!init)
        {
            return false;
        }

        if (init.getType().isa<mlir_ts::ConstArrayType>() || init.getType().isa<mlir_ts::ConstTupleType>())
        {
            return true;
        }

        auto defOp = init.getDefiningOp();
        if (isa<mlir_ts::ConstantOp>(defOp) || isa<mlir_ts::UndefOp>(defOp) || isa<mlir_ts::NullOp>(defOp))
        {
            return true;
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! is it const? : " << init << "\n";);

        return false;
    }

    bool registerVariable(mlir::Location location, StringRef name, bool isFullName, VariableClass varClass,
                          std::function<std::pair<mlir::Type, mlir::Value>()> func, const GenContext &genContext)
    {
        auto isGlobalScope = isFullName || !genContext.funcOp; /*symbolTable.getCurScope()->getParentScope() == nullptr*/
        auto isGlobal = isGlobalScope || varClass == VariableClass::Var;
        auto isConst = (varClass == VariableClass::Const || varClass == VariableClass::ConstRef) &&
                       !genContext.allocateVarsOutsideOfOperation && !genContext.allocateVarsInContextThis;

        auto effectiveName = name;

        mlir::Value variableOp;
        mlir::Type varType;
        if (!isGlobal)
        {
            auto res = func();
            auto type = std::get<0>(res);
            auto init = std::get<1>(res);
            if (!type && genContext.allowPartialResolve)
            {
                return false;
            }

            assert(type);
            varType = type;

            if (isConst)
            {
                variableOp = init;
                // special cast to support ForOf
                if (varClass == VariableClass::ConstRef)
                {
                    MLIRCodeLogic mcl(builder);
                    variableOp = mcl.GetReferenceOfLoadOp(init);
                    if (!variableOp)
                    {
                        // convert ConstRef to Const again as this is const object (it seems)
                        variableOp = init;
                        varClass = VariableClass::Const;
                    }
                }
            }
            else
            {
                assert(type);

                MLIRTypeHelper mth(builder.getContext());

                auto actualType = mth.convertConstArrayTypeToArrayType(type);

                // this is 'let', if 'let' is func, it should be HybridFunction
                if (auto funcType = actualType.dyn_cast<mlir_ts::FunctionType>())
                {
                    actualType = mlir_ts::HybridFunctionType::get(builder.getContext(), funcType);
                }

                if (init && actualType != type)
                {
                    auto castValue = cast(location, actualType, init, genContext);
                    init = castValue;
                }

                varType = actualType;

                // scope to restore inserting point
                {
                    mlir::OpBuilder::InsertionGuard insertGuard(builder);
                    if (genContext.allocateVarsOutsideOfOperation)
                    {
                        builder.setInsertionPoint(genContext.currentOperation);
                    }

                    if (genContext.allocateVarsInContextThis)
                    {
                        variableOp = registerVariableInThisContext(location, name, actualType, genContext);
                    }

                    if (!variableOp)
                    {
                        // default case
                        variableOp = builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(actualType),
                                                                         genContext.allocateVarsOutsideOfOperation ? mlir::Value() : init,
                                                                         builder.getBoolAttr(false));
                    }
                }
            }

            // init must be in its normal place
            if ((genContext.allocateVarsInContextThis || genContext.allocateVarsOutsideOfOperation) && variableOp && init && !isConst)
            {
                builder.create<mlir_ts::StoreOp>(location, init, variableOp);
            }
        }
        else
        {
            mlir_ts::GlobalOp globalOp;
            // get constant
            {
                mlir::OpBuilder::InsertionGuard insertGuard(builder);
                builder.setInsertionPointToStart(theModule.getBody());
                // find last string
                auto lastUse = [&](mlir::Operation *op) {
                    if (auto globalOp = dyn_cast_or_null<mlir_ts::GlobalOp>(op))
                    {
                        builder.setInsertionPointAfter(globalOp);
                    }
                };

                theModule.getBody()->walk(lastUse);

                effectiveName = getFullNamespaceName(name);

                globalOp = builder.create<mlir_ts::GlobalOp>(location,
                                                             // temp type
                                                             builder.getI32Type(), isConst, effectiveName, mlir::Attribute());

                if (isGlobalScope)
                {
                    auto &region = globalOp.getInitializerRegion();
                    auto *block = builder.createBlock(&region);

                    builder.setInsertionPoint(block, block->begin());

                    auto res = func();
                    auto type = std::get<0>(res);
                    auto init = std::get<1>(res);
                    if (!type && genContext.allowPartialResolve)
                    {
                        return false;
                    }

                    assert(type);
                    varType = type;

                    globalOp.typeAttr(mlir::TypeAttr::get(type));

                    // add return
                    // TODO: allow only ConstantOp or Undef or Null
                    if (init)
                    {
                        builder.create<mlir_ts::GlobalResultOp>(location, mlir::ValueRange{init});
                    }
                    else
                    {
                        auto undef = builder.create<mlir_ts::UndefOp>(location, type);
                        builder.create<mlir_ts::GlobalResultOp>(location, mlir::ValueRange{undef});
                    }
                }
            }

            if (!isGlobalScope)
            {
                auto res = func();
                auto type = std::get<0>(res);
                auto init = std::get<1>(res);
                if (!type && genContext.allowPartialResolve)
                {
                    return false;
                }

                assert(type);
                varType = type;

                globalOp.typeAttr(mlir::TypeAttr::get(type));

                if (init)
                {
                    // save value
                    auto address = builder.create<mlir_ts::AddressOfOp>(location, mlir_ts::RefType::get(type), name, mlir::IntegerAttr());
                    builder.create<mlir_ts::StoreOp>(location, init, address);
                }
            }
        }

#ifndef NDEBUG
        if (variableOp)
        {
            LLVM_DEBUG(dbgs() << "\n!! variable = " << effectiveName << " type: " << varType << " op: " << variableOp << "\n";);
        }
#endif

        auto varDecl = std::make_shared<VariableDeclarationDOM>(effectiveName, varType, location);
        if (!isConst || varClass == VariableClass::ConstRef)
        {
            varDecl->setReadWriteAccess();
        }

        varDecl->setFuncOp(genContext.funcOp);

        if (!isGlobal)
        {
            declare(varDecl, variableOp, genContext);
        }
        else if (isFullName)
        {
            fullNameGlobalsMap.insert(name, varDecl);
        }
        else
        {
            getGlobalsMap().insert({name, varDecl});
        }

        return true;
    }

    bool processDeclarationArrayBindingPattern(mlir::Location location, ArrayBindingPattern arrayBindingPattern, VariableClass varClass,
                                               std::function<std::pair<mlir::Type, mlir::Value>()> func, const GenContext &genContext)
    {
        auto res = func();
        auto type = std::get<0>(res);
        auto init = std::get<1>(res);

        auto index = 0;
        for (auto arrayBindingElement : arrayBindingPattern->elements)
        {
            MLIRPropertyAccessCodeLogic cl(builder, location, init, builder.getI32IntegerAttr(index));
            mlir::Value subInit;
            TypeSwitch<mlir::Type>(type)
                .template Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { subInit = cl.Tuple(constTupleType, true); })
                .template Case<mlir_ts::TupleType>([&](auto tupleType) { subInit = cl.Tuple(tupleType, true); })
                .template Case<mlir_ts::ConstArrayType>([&](auto constArrayType) {
                    // TODO: unify it with ElementAccess
                    auto constIndex = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(index));
                    auto elemRef = builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(constArrayType.getElementType()),
                                                                         init, constIndex);
                    subInit = builder.create<mlir_ts::LoadOp>(location, constArrayType.getElementType(), elemRef);
                })
                .template Case<mlir_ts::ArrayType>([&](auto tupleType) {
                    // TODO: unify it with ElementAccess
                    auto constIndex = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(index));
                    auto elemRef = builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(tupleType.getElementType()), init,
                                                                         constIndex);
                    subInit = builder.create<mlir_ts::LoadOp>(location, tupleType.getElementType(), elemRef);
                })
                .Default([&](auto type) { llvm_unreachable("not implemented"); });

            if (!processDeclaration(
                    arrayBindingElement.as<BindingElement>(), varClass, [&]() { return std::make_pair(subInit.getType(), subInit); },
                    genContext))
            {
                return false;
            }

            index++;
        }

        return true;
    }

    bool processDeclarationObjectBindingPattern(mlir::Location location, ObjectBindingPattern objectBindingPattern, VariableClass varClass,
                                                std::function<std::pair<mlir::Type, mlir::Value>()> func, const GenContext &genContext)
    {
        auto res = func();
        auto type = std::get<0>(res);
        auto init = std::get<1>(res);

        auto index = 0;
        for (auto objectBindingElement : objectBindingPattern->elements)
        {
            if (objectBindingElement->name == SyntaxKind::ObjectBindingPattern)
            {
                // nested obj, objectBindingElement->propertyName -> name
                auto name = MLIRHelper::getName(objectBindingElement->propertyName);
                auto subInit = mlirGenPropertyAccessExpression(location, init, name, false, genContext);

                auto objectBindingPattern = objectBindingElement->name.as<ObjectBindingPattern>();
                return processDeclarationObjectBindingPattern(
                    location, objectBindingPattern, varClass, [&]() { return std::make_pair(subInit.getType(), subInit); }, genContext);
            }

            auto name = MLIRHelper::getName(objectBindingElement->name);
            auto subInit = mlirGenPropertyAccessExpression(location, init, name, false, genContext);

            if (!processDeclaration(
                    objectBindingElement, varClass, [&]() { return std::make_pair(subInit.getType(), subInit); }, genContext))
            {
                return false;
            }

            index++;
        }

        return true;
    }

    bool processDeclaration(NamedDeclaration item, VariableClass varClass, std::function<std::pair<mlir::Type, mlir::Value>()> func,
                            const GenContext &genContext)
    {
        auto location = loc(item);

        if (item->name == SyntaxKind::ArrayBindingPattern)
        {
            auto arrayBindingPattern = item->name.as<ArrayBindingPattern>();
            if (!processDeclarationArrayBindingPattern(location, arrayBindingPattern, varClass, func, genContext))
            {
                return false;
            }
        }
        else if (item->name == SyntaxKind::ObjectBindingPattern)
        {
            auto objectBindingPattern = item->name.as<ObjectBindingPattern>();
            if (!processDeclarationObjectBindingPattern(location, objectBindingPattern, varClass, func, genContext))
            {
                return false;
            }
        }
        else
        {
            // name
            auto name = MLIRHelper::getName(item->name);

            // register
            return registerVariable(location, name, false, varClass, func, genContext);
        }

        return true;
    }

    template <typename ItemTy>
    std::pair<mlir::Type, mlir::Value> getTypeOnly(ItemTy item, mlir::Type defaultType, const GenContext &genContext)
    {
        // type
        mlir::Type type = defaultType;
        if (item->type)
        {
            type = getType(item->type, genContext);
        }

        return std::make_pair(type, mlir::Value());
    }

    template <typename ItemTy> std::pair<mlir::Type, bool> evaluateTypeAndInit(ItemTy item, const GenContext &genContext)
    {
        // type
        auto hasInit = false;
        mlir::Type type;
        if (item->type)
        {
            type = getType(item->type, genContext);
        }

        // init
        if (auto initializer = item->initializer)
        {
            hasInit = true;
            auto initType = evaluate(initializer, genContext);
            if (initType && !type)
            {
                type = initType;
            }
        }

        return std::make_pair(type, hasInit);
    }

    template <typename ItemTy> std::pair<mlir::Type, mlir::Value> getTypeAndInit(ItemTy item, const GenContext &genContext)
    {
        // type
        mlir::Type type;
        if (item->type)
        {
            type = getType(item->type, genContext);
        }

        // init
        mlir::Value init;
        if (auto initializer = item->initializer)
        {
            init = mlirGen(initializer, genContext);
            if (init)
            {
                if (!type)
                {
                    type = init.getType();
                }
                else if (type != init.getType())
                {
                    auto castValue = cast(loc(initializer), type, init, genContext);
                    init = castValue;
                }
            }
        }

        return std::make_pair(type, init);
    }

    mlir::LogicalResult mlirGen(VariableDeclaration item, VariableClass varClass, const GenContext &genContext)
    {
        if (!item->type && !item->initializer)
        {
            auto name = MLIRHelper::getName(item->name);
            emitError(loc(item)) << "type of variable '" << name << "' is not provided, variable must have type or initializer";
            return mlir::failure();
        }

        auto initFunc = [&]() { return getTypeAndInit(item, genContext); };

        auto valClassItem = varClass;
        if ((item->transformFlags & TransformFlags::ForceConst) == TransformFlags::ForceConst)
        {
            valClassItem = VariableClass::Const;
        }

        if ((item->transformFlags & TransformFlags::ForceConstRef) == TransformFlags::ForceConstRef)
        {
            valClassItem = VariableClass::ConstRef;
        }

        if (!processDeclaration(item, valClassItem, initFunc, genContext))
        {
            return mlir::failure();
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(VariableDeclarationList variableDeclarationListAST, const GenContext &genContext)
    {
        auto isLet = (variableDeclarationListAST->flags & NodeFlags::Let) == NodeFlags::Let;
        auto isConst = (variableDeclarationListAST->flags & NodeFlags::Const) == NodeFlags::Const;
        auto varClass = isLet ? VariableClass::Let : isConst ? VariableClass::Const : VariableClass::Var;

        for (auto &item : variableDeclarationListAST->declarations)
        {
            if (mlir::failed(mlirGen(item, varClass, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(VariableStatement variableStatementAST, const GenContext &genContext)
    {
        return mlirGen(variableStatementAST->declarationList, genContext);
    }

    std::pair<mlir::LogicalResult, std::vector<std::shared_ptr<FunctionParamDOM>>> mlirGenParameters(
        SignatureDeclarationBase parametersContextAST, const GenContext &genContext)
    {
        std::vector<std::shared_ptr<FunctionParamDOM>> params;

        // add this param
        auto isStatic = hasModifier(parametersContextAST, SyntaxKind::StaticKeyword);
        if (!isStatic && (parametersContextAST == SyntaxKind::MethodDeclaration || parametersContextAST == SyntaxKind::Constructor ||
                          parametersContextAST == SyntaxKind::GetAccessor || parametersContextAST == SyntaxKind::SetAccessor))
        {
            params.push_back(std::make_shared<FunctionParamDOM>(THIS_NAME, genContext.thisType, loc(parametersContextAST)));
        }

        if (!isStatic && genContext.thisType &&
            (parametersContextAST == SyntaxKind::FunctionExpression || parametersContextAST == SyntaxKind::ArrowFunction))
        {
            params.push_back(std::make_shared<FunctionParamDOM>(THIS_NAME, genContext.thisType, loc(parametersContextAST)));
        }

        if (parametersContextAST->parent.is<InterfaceDeclaration>())
        {
            params.push_back(std::make_shared<FunctionParamDOM>(THIS_NAME, getOpaqueType(), loc(parametersContextAST)));
        }

        auto formalParams = parametersContextAST->parameters;
        auto index = 0;
        for (auto arg : formalParams)
        {
            auto isBindingPattern = false;
            auto namePtr = MLIRHelper::getName(arg->name, stringAllocator);
            if (namePtr.empty())
            {
                isBindingPattern = true;

                std::stringstream ss;
                ss << "arg" << index;
                namePtr = mlir::StringRef(ss.str()).copy(stringAllocator);
            }

            mlir::Type type;
            auto isMultiArgs = !!arg->dotDotDotToken;
            auto isOptional = !!arg->questionToken;
            auto typeParameter = arg->type;
            if (typeParameter)
            {
                type = getType(typeParameter, genContext);
            }

            // process init value
            auto initializer = arg->initializer;
            if (initializer)
            {
                auto evalType = evaluate(initializer, genContext);
                if (evalType)
                {
                    // TODO: set type if not provided
                    isOptional = true;
                    if (isNoneType(type))
                    {
                        type = evalType;
                    }
                }
            }

            if (isNoneType(type) && genContext.argTypeDestFuncType)
            {
                if (auto funcType = genContext.argTypeDestFuncType.dyn_cast<mlir_ts::FunctionType>())
                {
                    type = funcType.getInput(index);
                }
                else if (auto hybridFuncType = genContext.argTypeDestFuncType.dyn_cast<mlir_ts::HybridFunctionType>())
                {
                    type = hybridFuncType.getInput(index);
                }

                LLVM_DEBUG(dbgs() << "\n!! param " << namePtr << " mapped to type " << type << "\n\n");
            }

            if (isNoneType(type))
            {
                if (!typeParameter && !initializer)
                {
                    auto funcName = MLIRHelper::getName(parametersContextAST->name);
                    emitError(loc(arg)) << "type of parameter '" << namePtr
                                        << "' is not provided, parameter must have type or initializer, function: " << funcName;
                    return {mlir::failure(), params};
                }

                emitError(loc(typeParameter)) << "can't resolve type for parameter '" << namePtr << "'";

                return {mlir::failure(), params};
            }

            /*
            if (!type)
            {
                // TODO: this is begginging of generic types
                type = getGenericType();
            }
            */

            if (isBindingPattern)
            {
                params.push_back(
                    std::make_shared<FunctionParamDOM>(namePtr, type, loc(arg), isOptional, isMultiArgs, initializer, arg->name));
            }
            else
            {
                params.push_back(std::make_shared<FunctionParamDOM>(namePtr, type, loc(arg), isOptional, isMultiArgs, initializer));
            }

            index++;
        }

        return {mlir::success(), params};
    }

    std::tuple<std::string, std::string> getNameOfFunction(SignatureDeclarationBase signatureDeclarationBaseAST,
                                                           const GenContext &genContext)
    {
        std::string fullName = getNameWithArguments(signatureDeclarationBaseAST, genContext);
        std::string objectOwnerName;
        if (signatureDeclarationBaseAST->parent.is<ClassDeclaration>())
        {
            objectOwnerName = getNameWithArguments(signatureDeclarationBaseAST->parent.as<ClassDeclaration>(), genContext);
        }
        else if (signatureDeclarationBaseAST->parent.is<InterfaceDeclaration>())
        {
            objectOwnerName = getNameWithArguments(signatureDeclarationBaseAST->parent.as<InterfaceDeclaration>(), genContext);
        }
        else if (genContext.funcOp)
        {
            auto funcName = const_cast<GenContext &>(genContext).funcOp.sym_name().str();
            objectOwnerName = funcName;
        }

        if (signatureDeclarationBaseAST == SyntaxKind::MethodDeclaration)
        {
            if (!genContext.thisType.isa<mlir_ts::ObjectType>())
            {
                // class method name
                fullName = objectOwnerName + "." + fullName;
            }
            else
            {
                fullName = "";
            }
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::MethodSignature)
        {
            // class method name
            fullName = objectOwnerName + "." + fullName;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::GetAccessor)
        {
            // class method name
            fullName = objectOwnerName + ".get_" + fullName;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::SetAccessor)
        {
            // class method name
            fullName = objectOwnerName + ".set_" + fullName;
        }
        else if (signatureDeclarationBaseAST == SyntaxKind::Constructor)
        {
            // class method name
            auto isStatic = hasModifier(signatureDeclarationBaseAST, SyntaxKind::StaticKeyword);
            if (isStatic)
            {
                fullName = objectOwnerName + "." + STATIC_CONSTRUCTOR_NAME;
            }
            else
            {
                fullName = objectOwnerName + "." + CONSTRUCTOR_NAME;
            }
        }

        auto name = fullName;
        if (fullName.empty())
        {
            // auto calculate name
            name = fullName = MLIRHelper::getAnonymousName(loc_check(signatureDeclarationBaseAST));
        }
        else
        {
            fullName = getFullNamespaceName(fullName).str();
        }

        return std::make_tuple(fullName, name);
    }

    std::tuple<FunctionPrototypeDOM::TypePtr, mlir_ts::FunctionType, SmallVector<mlir::Type>> mlirGenFunctionSignaturePrototype(
        SignatureDeclarationBase signatureDeclarationBaseAST, bool defaultVoid, const GenContext &genContext)
    {
        auto res = getNameOfFunction(signatureDeclarationBaseAST, genContext);
        auto fullName = std::get<0>(res);
        auto name = std::get<1>(res);

        mlir_ts::FunctionType funcType;
        auto paramsResult = mlirGenParameters(signatureDeclarationBaseAST, genContext);
        auto result = std::get<0>(paramsResult);
        if (mlir::failed(result))
        {
            return std::make_tuple(FunctionPrototypeDOM::TypePtr(nullptr), funcType, SmallVector<mlir::Type>{});
        }

        auto params = std::get<1>(paramsResult);
        SmallVector<mlir::Type> argTypes;
        auto argNumber = 0;
        auto isMultiArgs = false;

        // auto isAsync = hasModifier(signatureDeclarationBaseAST, SyntaxKind::AsyncKeyword);

        for (const auto &param : params)
        {
            auto paramType = param->getType();
            if (isNoneType(paramType))
            {
                return std::make_tuple(FunctionPrototypeDOM::TypePtr(nullptr), funcType, SmallVector<mlir::Type>{});
            }

            if (param->getIsOptional() && !paramType.isa<mlir_ts::OptionalType>())
            {
                argTypes.push_back(getOptionalType(paramType));
            }
            else
            {
                argTypes.push_back(paramType);
            }

            isMultiArgs |= param->getIsMultiArgs();

            argNumber++;
        }

        auto funcProto = std::make_shared<FunctionPrototypeDOM>(fullName, params);

        funcProto->setNameWithoutNamespace(name);

        // check if function already discovered
        auto funcIt = getFunctionMap().find(name);
        if (funcIt != getFunctionMap().end())
        {
            auto cachedFuncType = funcIt->second.getType();
            if (cachedFuncType.getNumResults() > 0)
            {
                auto returnType = cachedFuncType.getResult(0);
                funcProto->setReturnType(returnType);
            }

            funcType = cachedFuncType;
        }
        else if (auto typeParameter = signatureDeclarationBaseAST->type)
        {
            auto returnType = getType(typeParameter, genContext);
            funcProto->setReturnType(returnType);

            funcType = getFunctionType(argTypes, returnType, isMultiArgs);
        }
        else if (defaultVoid)
        {
            auto returnType = getVoidType();
            funcProto->setReturnType(returnType);

            funcType = getFunctionType(argTypes, returnType, isMultiArgs);
        }

        return std::make_tuple(funcProto, funcType, argTypes);
    }

    std::tuple<mlir_ts::FuncOp, FunctionPrototypeDOM::TypePtr, bool> mlirGenFunctionPrototype(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        auto location = loc(functionLikeDeclarationBaseAST);

        mlir_ts::FuncOp funcOp;

        auto res = mlirGenFunctionSignaturePrototype(functionLikeDeclarationBaseAST, false, genContext);
        auto funcProto = std::get<0>(res);
        if (!funcProto)
        {
            return std::make_tuple(funcOp, funcProto, false);
        }

        auto funcType = std::get<1>(res);
        auto argTypes = std::get<2>(res);
        auto fullName = funcProto->getName();

        // discover type & args
        // !genContext.allowPartialResolve -> in actual process we need actual data
        if (!funcType || genContext.rediscover)
        {
            if (mlir::succeeded(
                    discoverFunctionReturnTypeAndCapturedVars(functionLikeDeclarationBaseAST, fullName, argTypes, funcProto, genContext)))
            {
                // rewrite ret type with actual value
                if (auto typeParameter = functionLikeDeclarationBaseAST->type)
                {
                    auto returnType = getType(typeParameter, genContext);
                    funcProto->setReturnType(returnType);
                }
                else if (genContext.argTypeDestFuncType)
                {
                    auto &argTypeDestFuncType = genContext.argTypeDestFuncType;
                    if (auto funcType = argTypeDestFuncType.dyn_cast<mlir_ts::FunctionType>())
                    {
                        if (funcType.getNumResults() > 0)
                        {
                            funcProto->setReturnType(funcType.getResult(0));
                        }
                    }
                    else if (auto hybridFuncType = argTypeDestFuncType.dyn_cast<mlir_ts::HybridFunctionType>())
                    {
                        if (hybridFuncType.getResults().size() > 0)
                        {
                            funcProto->setReturnType(hybridFuncType.getResult(0));
                        }
                    }
                    else if (auto boundFuncType = argTypeDestFuncType.dyn_cast<mlir_ts::BoundFunctionType>())
                    {
                        if (boundFuncType.getResults().size() > 0)
                        {
                            funcProto->setReturnType(boundFuncType.getResult(0));
                        }
                    }
                }

                // create funcType
                if (funcProto->getReturnType())
                {
                    funcType = getFunctionType(argTypes, funcProto->getReturnType(), funcProto->isMultiArgs());
                }
                else
                {
                    // no return type
                    funcType = getFunctionType(argTypes, llvm::None, funcProto->isMultiArgs());
                }
            }
            else
            {
                // false result
                return std::make_tuple(funcOp, funcProto, false);
            }
        }

        // we need it, when we run rediscovery second time
        if (!funcProto->getHasExtraFields())
        {
            funcProto->setHasExtraFields(getLocalVarsInThisContextMap().find(funcProto->getName()) != getLocalVarsInThisContextMap().end());
        }

        auto it = getCaptureVarsMap().find(funcProto->getName());
        auto hasCapturedVars = funcProto->getHasCapturedVars() || (it != getCaptureVarsMap().end());
        if (hasCapturedVars)
        {
            // important set when it is discovered and in process second type
            funcProto->setHasCapturedVars(true);

#ifndef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION

            SmallVector<mlir::NamedAttribute> attrs;
            SmallVector<mlir::DictionaryAttr> argAttrs;

#ifdef ADD_GC_ATTRIBUTE
            attrs.push_back({builder.getIdentifier(TS_GC_ATTRIBUTE), mlir::UnitAttr::get(builder.getContext())});
#endif

            for (auto argType : funcType.getInputs())
            {
                SmallVector<mlir::NamedAttribute> argAttrsForType;
                // add nested to first attr
                if (argAttrs.size() == 0)
                {
                    // we need to force LLVM converter to allow to amend op in attached interface
                    attrs.push_back({builder.getIdentifier(TS_NEST_ATTRIBUTE), mlir::UnitAttr::get(builder.getContext())});
                    argAttrsForType.push_back({builder.getIdentifier(TS_NEST_ATTRIBUTE), mlir::UnitAttr::get(builder.getContext())});
                }

                auto argDicAttr = mlir::DictionaryAttr::get(builder.getContext(), argAttrsForType);
                argAttrs.push_back(argDicAttr);
            }

            funcOp = mlir_ts::FuncOp::create(location, fullName, funcType, attrs, argAttrs);
#else
            funcOp = mlir_ts::FuncOp::create(location, fullName, funcType);
#endif
        }
        else
        {
#ifdef ADD_GC_ATTRIBUTE
            SmallVector<mlir::NamedAttribute> attrs;
            attrs.push_back({builder.getIdentifier(TS_GC_ATTRIBUTE), mlir::UnitAttr::get(builder.getContext())});
            funcOp = mlir_ts::FuncOp::create(location, fullName, funcType, attrs);
#else
            funcOp = mlir_ts::FuncOp::create(location, fullName, funcType);
#endif
        }

        return std::make_tuple(funcOp, funcProto, true);
    }

    mlir::LogicalResult discoverFunctionReturnTypeAndCapturedVars(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST,
                                                                  StringRef name, SmallVector<mlir::Type> &argTypes,
                                                                  const FunctionPrototypeDOM::TypePtr &funcProto,
                                                                  const GenContext &genContext)
    {
        if (funcProto->getDiscovered())
        {
            return mlir::failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! discovering 'ret type' & 'captured vars' for : " << name << "\n";);

        mlir::OpBuilder::InsertionGuard guard(builder);

        auto partialDeclFuncType = getFunctionType(argTypes, llvm::None);
        auto dummyFuncOp = mlir_ts::FuncOp::create(loc(functionLikeDeclarationBaseAST), name, partialDeclFuncType);

        {
            // simulate scope
            SymbolTableScopeT varScope(symbolTable);

            GenContext genContextWithPassResult{};
            genContextWithPassResult.funcOp = dummyFuncOp;
            genContextWithPassResult.thisType = genContext.thisType;
            genContextWithPassResult.allowPartialResolve = true;
            genContextWithPassResult.dummyRun = true;
            genContextWithPassResult.cleanUps = new SmallVector<mlir::Block *>();
            genContextWithPassResult.passResult = new PassResult();
            genContextWithPassResult.state = new int(1);
            genContextWithPassResult.allocateVarsInContextThis =
                (functionLikeDeclarationBaseAST->transformFlags & TransformFlags::VarsInObjectContext) ==
                TransformFlags::VarsInObjectContext;
            genContextWithPassResult.unresolved = genContext.unresolved;

            registerNamespace(funcProto->getNameWithoutNamespace(), true);

            if (succeeded(mlirGenFunctionBody(functionLikeDeclarationBaseAST, dummyFuncOp, funcProto, genContextWithPassResult)))
            {
                exitNamespace();

                auto &passResult = genContextWithPassResult.passResult;
                if (!passResult->functionReturnType && passResult->functionReturnTypeShouldBeProvided)
                {
                    // has return value but type is not provided yet
                    genContextWithPassResult.clean();
                    return mlir::failure();
                }

                funcProto->setDiscovered(true);
                auto discoveredType = passResult->functionReturnType;
                if (discoveredType && discoveredType != funcProto->getReturnType())
                {
                    // TODO: do we need to convert it here? maybe send it as const object?
                    MLIRTypeHelper mth(builder.getContext());
                    funcProto->setReturnType(mth.convertConstArrayTypeToArrayType(discoveredType));
                    LLVM_DEBUG(llvm::dbgs() << "\n!! ret type: " << funcProto->getReturnType() << ", name: " << name << "\n";);
                }

                // if we have captured parameters, add first param to send lambda's type(class)
                if (passResult->outerVariables.size() > 0)
                {
                    MLIRCodeLogic mcl(builder);
#ifdef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
                    auto isObjectType = genContext.thisType != nullptr && genContext.thisType.isa<mlir_ts::ObjectType>();
                    if (!isObjectType)
                    {
#endif
                        argTypes.insert(argTypes.begin(), mcl.CaptureType(passResult->outerVariables));
#ifdef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
                    }
#endif
                    getCaptureVarsMap().insert({name, passResult->outerVariables});
                    funcProto->setHasCapturedVars(true);

                    LLVM_DEBUG(llvm::dbgs() << "\n!! has captured vars, name: " << name << "\n";);
                }

                if (passResult->extraFieldsInThisContext.size() > 0)
                {
                    getLocalVarsInThisContextMap().insert({name, passResult->extraFieldsInThisContext});

                    funcProto->setHasExtraFields(true);
                }

                genContextWithPassResult.clean();
                return mlir::success();
            }
            else
            {
                exitNamespace();

                genContextWithPassResult.clean();
                return mlir::failure();
            }
        }
    }

    mlir::LogicalResult mlirGen(FunctionDeclaration functionDeclarationAST, const GenContext &genContext)
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        auto res = mlirGenFunctionLikeDeclaration(functionDeclarationAST, genContext);
        return std::get<0>(res);
    }

    mlir::Value mlirGen(FunctionExpression functionExpressionAST, const GenContext &genContext)
    {
        auto location = loc(functionExpressionAST);
        mlir_ts::FuncOp funcOp;

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.restoreInsertionPoint(functionBeginPoint);

            // provide name for it
            auto funcGenContext = GenContext(genContext);
            funcGenContext.thisType = nullptr;
            auto res = mlirGenFunctionLikeDeclaration(functionExpressionAST, funcGenContext);
            if (mlir::failed(std::get<0>(res)))
            {
                return mlir::Value();
            }

            funcOp = std::get<1>(res);
        }

        if (auto trampOp = resolveFunctionWithCapture(location, funcOp.getName(), funcOp.getType(), false, genContext))
        {
            return trampOp;
        }

        auto funcSymbolRef = builder.create<mlir_ts::SymbolRefOp>(loc(functionExpressionAST), funcOp.getType(),
                                                                  mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
        return funcSymbolRef;
    }

    mlir::Value mlirGen(ArrowFunction arrowFunctionAST, const GenContext &genContext)
    {
        auto location = loc(arrowFunctionAST);
        mlir_ts::FuncOp funcOp;

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.restoreInsertionPoint(functionBeginPoint);

            // provide name for it
            auto allowFuncGenContext = GenContext(genContext);
            allowFuncGenContext.thisType = nullptr;
            auto res = mlirGenFunctionLikeDeclaration(arrowFunctionAST, allowFuncGenContext);
            if (mlir::failed(std::get<0>(res)))
            {
                return mlir::Value();
            }

            funcOp = std::get<1>(res);
        }

        if (auto trampOp = resolveFunctionWithCapture(location, funcOp.getName(), funcOp.getType(), false, genContext))
        {
            return trampOp;
        }

        auto funcSymbolRef = builder.create<mlir_ts::SymbolRefOp>(location, funcOp.getType(),
                                                                  mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
        return funcSymbolRef;
    }

    std::pair<mlir::LogicalResult, mlir_ts::FuncOp> mlirGenFunctionGenerator(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST,
                                                                             const GenContext &genContext)
    {
        auto location = loc(functionLikeDeclarationBaseAST);
        NodeFactory nf(NodeFactoryFlags::None);

        auto stepIdent = nf.createIdentifier(S("step"));

        // create return object
        NodeArray<ObjectLiteralElementLike> generatorObjectProperties;

        // add step field
        auto stepProp = nf.createPropertyAssignment(stepIdent, nf.createNumericLiteral(S("0"), TokenFlags::None));
        generatorObjectProperties.push_back(stepProp);

        // create body of next method
        NodeArray<Statement> nextStatements;

        // add main switcher
        auto stepAccess = nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword), stepIdent);

        // call stateswitch
        NodeArray<Expression> args;
        args.push_back(stepAccess);
        auto callStat = nf.createExpressionStatement(nf.createCallExpression(nf.createIdentifier(S("switchstate")), undefined, args));

        nextStatements.push_back(callStat);

        // add function body to statements to first step
        if (functionLikeDeclarationBaseAST->body == SyntaxKind::Block)
        {
            // process every statement
            auto block = functionLikeDeclarationBaseAST->body.as<Block>();
            for (auto statement : block->statements)
            {
                nextStatements.push_back(statement);
            }
        }
        else
        {
            nextStatements.push_back(functionLikeDeclarationBaseAST->body);
        }

        // add next statements
        // add default return with empty
        nextStatements.push_back(nf.createReturnStatement(getYieldReturnObject(nf, nf.createIdentifier(S("undefined")), true)));

        // create next body
        auto nextBody = nf.createBlock(nextStatements, /*multiLine*/ false);

        // create method next in object
        auto nextMethodDecl = nf.createMethodDeclaration(undefined, undefined, undefined, nf.createIdentifier(S("next")), undefined,
                                                         undefined, undefined, undefined, nextBody);
        nextMethodDecl->transformFlags |= TransformFlags::VarsInObjectContext;

        // copy location info, to fix issue with names of anonymous functions
        nextMethodDecl->pos = functionLikeDeclarationBaseAST->pos;
        nextMethodDecl->_end = functionLikeDeclarationBaseAST->_end;

        generatorObjectProperties.push_back(nextMethodDecl);

        auto generatorObject = nf.createObjectLiteralExpression(generatorObjectProperties, false);

        // generator body
        NodeArray<Statement> generatorStatements;

        // step 1, add return object
        auto retStat = nf.createReturnStatement(generatorObject);
        generatorStatements.push_back(retStat);

        auto body = nf.createBlock(generatorStatements, /*multiLine*/ false);
        auto funcOp =
            nf.createFunctionDeclaration(functionLikeDeclarationBaseAST->decorators, functionLikeDeclarationBaseAST->modifiers, undefined,
                                         functionLikeDeclarationBaseAST->name, functionLikeDeclarationBaseAST->typeParameters,
                                         functionLikeDeclarationBaseAST->parameters, functionLikeDeclarationBaseAST->type, body);

        auto genFuncOp = mlirGenFunctionLikeDeclaration(funcOp, genContext);
        return genFuncOp;
    }

    mlir::LogicalResult registerGenericFunctionLike(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST,
                                                    const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(functionLikeDeclarationBaseAST->name);
        if (!name.empty())
        {
            auto namePtr = StringRef(name).copy(stringAllocator);
            if (getGenericFunctionMap().count(namePtr))
            {
                return mlir::success();
            }

            llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
            if (mlir::failed(processTypeParameters(functionLikeDeclarationBaseAST->typeParameters, typeParameters, genContext)))
            {
                return mlir::failure();
            }

            // register class
            GenericFunctionInfo::TypePtr newGenericFunctionPtr = std::make_shared<GenericFunctionInfo>();
            newGenericFunctionPtr->name = namePtr;
            newGenericFunctionPtr->typeParams = typeParameters;
            newGenericFunctionPtr->functionDeclaration = functionLikeDeclarationBaseAST;

            getGenericFunctionMap().insert({namePtr, newGenericFunctionPtr});

            return mlir::success();
        }

        return mlir::failure();
    }

    std::pair<mlir::LogicalResult, mlir_ts::FuncOp> mlirGenFunctionLikeDeclaration(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        auto isGenericFunction = functionLikeDeclarationBaseAST->typeParameters.size() > 0;
        if (isGenericFunction && genContext.typeParamsWithArgs.size() == 0)
        {
            return {registerGenericFunctionLike(functionLikeDeclarationBaseAST, genContext), mlir_ts::FuncOp()};
        }

        // check if it is generator
        if (functionLikeDeclarationBaseAST->asteriskToken)
        {
            // this is generator, let's generate other function out of it
            return mlirGenFunctionGenerator(functionLikeDeclarationBaseAST, genContext);
        }

        // do not process generic functions more then 1 time
        if (isGenericFunction && genContext.typeParamsWithArgs.size() > 0)
        {
            auto functionName = getNameWithArguments(functionLikeDeclarationBaseAST, genContext);
            auto fn = getFunctionMap().find(functionName);
            if (fn != getFunctionMap().end() && theModule.lookupSymbol(functionName))
            {
                auto funcOp = fn->getValue();
                return {mlir::success(), funcOp};
            }
        }

        // go to root
        mlir::OpBuilder::InsertPoint savePoint;
        if (isGenericFunction)
        {
            savePoint = builder.saveInsertionPoint();
            builder.setInsertionPointToStart(&theModule.body().front());
        }

        auto location = loc(functionLikeDeclarationBaseAST);

        auto funcOpWithFuncProto = mlirGenFunctionPrototype(functionLikeDeclarationBaseAST, genContext);

        auto &funcOp = std::get<0>(funcOpWithFuncProto);
        auto &funcProto = std::get<1>(funcOpWithFuncProto);
        auto result = std::get<2>(funcOpWithFuncProto);
        if (!result || !funcOp)
        {
            return {mlir::failure(), funcOp};
        }

        auto funcGenContext = GenContext(genContext);
        funcGenContext.funcOp = funcOp;
        funcGenContext.passResult = nullptr;
        funcGenContext.state = new int(1);
        // if funcGenContext.passResult is null and allocateVarsInContextThis is true, this type should contain fully defined object with
        // local variables as fields
        funcGenContext.allocateVarsInContextThis =
            (functionLikeDeclarationBaseAST->transformFlags & TransformFlags::VarsInObjectContext) == TransformFlags::VarsInObjectContext;

        auto it = getCaptureVarsMap().find(funcProto->getName());
        if (it != getCaptureVarsMap().end())
        {
            funcGenContext.capturedVars = &it->getValue();
        }

        auto resultFromBody = mlir::failure();
        {
            registerNamespace(funcProto->getNameWithoutNamespace(), true);

            SymbolTableScopeT varScope(symbolTable);
            resultFromBody = mlirGenFunctionBody(functionLikeDeclarationBaseAST, funcOp, funcProto, funcGenContext);

            exitNamespace();
        }

        funcGenContext.cleanState();

        if (mlir::failed(resultFromBody))
        {
            return {mlir::failure(), funcOp};
        }

        // set visibility index
        if (funcProto->getName() != MAIN_ENTRY_NAME &&
            !hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::ExportKeyword) /* && !funcProto->getNoBody()*/)
        {
            funcOp.setPrivate();
        }

        if (!genContext.dummyRun)
        {
            theModule.push_back(funcOp);
        }

        auto name = funcProto->getNameWithoutNamespace();
        if (!getFunctionMap().count(name))
        {
            getFunctionMap().insert({name, funcOp});

            LLVM_DEBUG(llvm::dbgs() << "\n!! reg. func: " << name << " type:" << funcOp.getType() << "\n";);
            LLVM_DEBUG(llvm::dbgs() << "\n!! reg. func: " << name << " full name: " << funcProto->getName()
                                    << " num inputs:" << funcOp.getType().cast<mlir_ts::FunctionType>().getNumInputs() << "\n";);
        }
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! re-process. func: " << name << " type:" << funcOp.getType() << "\n";);
            LLVM_DEBUG(llvm::dbgs() << "\n!! re-process. func: " << name
                                    << " num inputs:" << funcOp.getType().cast<mlir_ts::FunctionType>().getNumInputs() << "\n";);
        }

        if (isGenericFunction)
        {
            builder.restoreInsertionPoint(savePoint);
        }
        else
        {
            builder.setInsertionPointAfter(funcOp);
        }

        return {mlir::success(), funcOp};
    }

    mlir::LogicalResult mlirGenFunctionEntry(mlir::Location location, FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
    {
        auto retType = funcProto->getReturnType();
        auto hasReturn = retType && !retType.isa<mlir_ts::VoidType>();
        if (hasReturn)
        {
            auto entryOp = builder.create<mlir_ts::EntryOp>(location, mlir_ts::RefType::get(retType));
            auto varDecl = std::make_shared<VariableDeclarationDOM>(RETURN_VARIABLE_NAME, retType, location);
            varDecl->setReadWriteAccess();
            declare(varDecl, entryOp.reference(), genContext);
        }
        else
        {
            builder.create<mlir_ts::EntryOp>(location, mlir::Type());
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionExit(mlir::Location location, const GenContext &genContext)
    {
        auto callableResult = const_cast<GenContext &>(genContext).funcOp.getCallableResults();
        auto retType = callableResult.size() > 0 ? callableResult.front() : mlir::Type();
        auto hasReturn = retType && !retType.isa<mlir_ts::VoidType>();
        if (hasReturn)
        {
            auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
            if (!retVarInfo.second)
            {
                if (genContext.allowPartialResolve)
                {
                    return mlir::success();
                }

                emitError(location) << "can't find return variable";
                return mlir::failure();
            }

            builder.create<mlir_ts::ExitOp>(location, retVarInfo.first);
        }
        else
        {
            builder.create<mlir_ts::ExitOp>(location, mlir::Value());
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionCapturedParam(mlir::Location loc, int &firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                                     mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        if (genContext.capturedVars == nullptr)
        {
            return mlir::success();
        }

#ifdef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
        auto isObjectType = genContext.thisType != nullptr && genContext.thisType.isa<mlir_ts::ObjectType>();
        if (isObjectType)
        {
            return mlir::success();
        }
#endif

        firstIndex++;

        auto capturedParam = arguments[firstIndex];
        auto capturedRefType = capturedParam.getType();

        auto capturedParamVar = std::make_shared<VariableDeclarationDOM>(CAPTURED_NAME, capturedRefType, loc);

        declare(capturedParamVar, capturedParam, genContext);

        return mlir::success();
    }

#ifdef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
    mlir::LogicalResult mlirGenFunctionCapturedParamIfObject(mlir::Location loc, int &firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                                             mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        if (genContext.capturedVars == nullptr)
        {
            return mlir::success();
        }

        auto isObjectType = genContext.thisType != nullptr && genContext.thisType.isa<mlir_ts::ObjectType>();
        if (isObjectType)
        {
            MLIRTypeHelper mth(builder.getContext());

            auto thisVal = resolveIdentifier(loc, THIS_NAME, genContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! this value: " << thisVal << "\n";);

            mlir::Value propValue = mlirGenPropertyAccessExpression(loc, thisVal, mth.TupleFieldName(CAPTURED_NAME), genContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! this->.captured value: " << propValue << "\n";);

            assert(propValue);

            // captured is in this->".captured"
            auto capturedParamVar = std::make_shared<VariableDeclarationDOM>(CAPTURED_NAME, propValue.getType(), loc);
            declare(capturedParamVar, propValue, genContext);
        }

        return mlir::success();
    }
#endif

    mlir::LogicalResult mlirGenFunctionParams(int firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                              mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        auto index = firstIndex;
        for (const auto &param : funcProto->getArgs())
        {
            index++;
            mlir::Value paramValue;

            // process init expression
            auto location = param->getLoc();

            // alloc all args
            // process optional parameters
            if (param->hasInitValue())
            {
                auto dataType = param->getType();
                auto paramOptionalOp = builder.create<mlir_ts::ParamOptionalOp>(location, mlir_ts::RefType::get(dataType), arguments[index],
                                                                                builder.getBoolAttr(false));

                paramValue = paramOptionalOp;

                /*auto *defValueBlock =*/builder.createBlock(&paramOptionalOp.defaultValueRegion());

                mlir::Value defaultValue;
                auto initExpression = param->getInitValue();
                if (initExpression)
                {
                    defaultValue = mlirGen(initExpression, genContext);
                }
                else
                {
                    llvm_unreachable("unknown statement");
                }

                if (defaultValue.getType() != dataType)
                {
                    defaultValue = cast(location, dataType, defaultValue, genContext);
                }

                builder.create<mlir_ts::ParamDefaultValueOp>(location, defaultValue);

                builder.setInsertionPointAfter(paramOptionalOp);
            }
            else if (param->getIsOptional() && !param->getType().isa<mlir_ts::OptionalType>())
            {
                auto optType = getOptionalType(param->getType());
                param->setType(optType);
                paramValue = builder.create<mlir_ts::ParamOp>(location, mlir_ts::RefType::get(optType), arguments[index],
                                                              builder.getBoolAttr(false));
            }
            else
            {
                paramValue = builder.create<mlir_ts::ParamOp>(location, mlir_ts::RefType::get(param->getType()), arguments[index],
                                                              builder.getBoolAttr(false));
            }

            if (paramValue)
            {
                // redefine variable
                param->setReadWriteAccess();
                declare(param, paramValue, genContext, true);
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionParamsBindings(int firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                                      mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        for (const auto &param : funcProto->getArgs())
        {
            if (auto bindingPattern = param->getBindingPattern())
            {
                auto location = loc(bindingPattern);
                auto val = resolveIdentifier(location, param->getName(), genContext);
                auto initFunc = [&]() { return std::make_pair(val.getType(), val); };

                if (bindingPattern == SyntaxKind::ArrayBindingPattern)
                {
                    auto arrayBindingPattern = bindingPattern.as<ArrayBindingPattern>();
                    if (!processDeclarationArrayBindingPattern(location, arrayBindingPattern, VariableClass::Let, initFunc, genContext))
                    {
                        continue;
                    }
                }
                else if (bindingPattern == SyntaxKind::ObjectBindingPattern)
                {
                    auto objectBindingPattern = bindingPattern.as<ObjectBindingPattern>();
                    if (!processDeclarationObjectBindingPattern(location, objectBindingPattern, VariableClass::Let, initFunc, genContext))
                    {
                        continue;
                    }
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionCaptures(FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
    {
        if (genContext.capturedVars == nullptr)
        {
            return mlir::success();
        }

        auto capturedVars = *genContext.capturedVars;

        NodeFactory nf(NodeFactoryFlags::None);

        // create variables
        for (auto &capturedVar : capturedVars)
        {
            auto varItem = capturedVar.getValue();
            auto variableInfo = varItem;
            auto name = variableInfo->getName();

            // load this.<var name>
            auto _captured = nf.createIdentifier(stows(CAPTURED_NAME));
            auto _name = nf.createIdentifier(stows(std::string(name)));
            auto _captured_name = nf.createPropertyAccessExpression(_captured, _name);
            auto capturedVarValue = mlirGen(_captured_name, genContext);
            auto variableRefType = mlir_ts::RefType::get(variableInfo->getType());

            auto capturedParam = std::make_shared<VariableDeclarationDOM>(name, variableRefType, variableInfo->getLoc());
            assert(capturedVarValue);
            if (capturedVarValue.getType().isa<mlir_ts::RefType>())
            {
                capturedParam->setReadWriteAccess();
            }

            LLVM_DEBUG(dbgs() << "\n!! captured '\".captured\"->" << name << "' [ " << capturedVarValue << " ] ref val type: [ "
                              << variableRefType << " ]\n\n");

            declare(capturedParam, capturedVarValue, genContext);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionBody(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, mlir_ts::FuncOp funcOp,
                                            FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
    {
        if (!functionLikeDeclarationBaseAST->body)
        {
            // it is just declaration
            funcProto->setNoBody(true);
            return mlir::success();
        }

        auto location = loc(functionLikeDeclarationBaseAST);

        auto *blockPtr = funcOp.addEntryBlock();
        auto &entryBlock = *blockPtr;

        // process function params
        for (auto paramPairs : llvm::zip(funcProto->getArgs(), entryBlock.getArguments()))
        {
            if (failed(declare(std::get<0>(paramPairs), std::get<1>(paramPairs), genContext)))
            {
                return mlir::failure();
            }
        }

        // allocate all params

        builder.setInsertionPointToStart(&entryBlock);

        auto arguments = entryBlock.getArguments();
        auto firstIndex = -1;

        // add exit code
        if (failed(mlirGenFunctionEntry(location, funcProto, genContext)))
        {
            return mlir::failure();
        }

        // register this if lambda function
        if (failed(mlirGenFunctionCapturedParam(location, firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

        // allocate function parameters as variable
        if (failed(mlirGenFunctionParams(firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenFunctionParamsBindings(firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

#ifdef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
        if (failed(mlirGenFunctionCapturedParamIfObject(location, firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }
#endif

        if (failed(mlirGenFunctionCaptures(funcProto, genContext)))
        {
            return mlir::failure();
        }

        if (failed(mlirGenBody(functionLikeDeclarationBaseAST->body, genContext)))
        {
            return mlir::failure();
        }

        // add exit code
        if (failed(mlirGenFunctionExit(location, genContext)))
        {
            return mlir::failure();
        }

        if (genContext.dummyRun)
        {
            genContext.cleanUps->push_back(blockPtr);
        }

        return mlir::success();
    }

    mlir::Value mlirGen(TypeAssertion typeAssertionAST, const GenContext &genContext)
    {
        auto location = loc(typeAssertionAST);

        auto typeInfo = getType(typeAssertionAST->type, genContext);
        auto exprValue = mlirGen(typeAssertionAST->expression, genContext);

        auto castedValue = cast(location, typeInfo, exprValue, genContext);
        return castedValue;
    }

    mlir::Value mlirGen(AsExpression asExpressionAST, const GenContext &genContext)
    {
        auto location = loc(asExpressionAST);

        auto typeInfo = getType(asExpressionAST->type, genContext);
        auto exprValue = mlirGen(asExpressionAST->expression, genContext);

        auto castedValue = cast(location, typeInfo, exprValue, genContext);
        return castedValue;
    }

    mlir::LogicalResult mlirGen(ReturnStatement returnStatementAST, const GenContext &genContext)
    {
        auto location = loc(returnStatementAST);
        if (auto expression = returnStatementAST->expression)
        {
            auto expressionValue = mlirGen(expression, genContext);
            return mlirGenReturnValue(location, expressionValue, false, genContext);
        }

        builder.create<mlir_ts::ReturnOp>(location);
        return mlir::success();
    }

    ObjectLiteralExpression getYieldReturnObject(NodeFactory &nf, Expression expr, bool stop)
    {
        auto valueIdent = nf.createIdentifier(S("value"));
        auto doneIdent = nf.createIdentifier(S("done"));

        NodeArray<ObjectLiteralElementLike> retObjectProperties;
        auto valueProp = nf.createPropertyAssignment(valueIdent, expr);
        retObjectProperties.push_back(valueProp);

        auto doneProp = nf.createPropertyAssignment(doneIdent, nf.createToken(stop ? SyntaxKind::TrueKeyword : SyntaxKind::FalseKeyword));
        retObjectProperties.push_back(doneProp);

        auto retObject = nf.createObjectLiteralExpression(retObjectProperties, stop);
        return retObject;
    };

    mlir::Value mlirGenYieldStar(YieldExpression yieldExpressionAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        NodeFactory nf(NodeFactoryFlags::None);

        auto _v_ident = nf.createIdentifier(S("_v_"));

        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_v_ident));
        auto declList = nf.createVariableDeclarationList(declarations, NodeFlags::Const);

        auto forOfStat = nf.createForOfStatement(undefined, declList, yieldExpressionAST->expression,
                                                 nf.createExpressionStatement(nf.createYieldExpression(undefined, _v_ident)));

        mlirGen(forOfStat, genContext);

        return mlir::Value();
    }

    mlir::Value mlirGen(YieldExpression yieldExpressionAST, const GenContext &genContext)
    {
        if (yieldExpressionAST->asteriskToken)
        {
            return mlirGenYieldStar(yieldExpressionAST, genContext);
        }

        auto location = loc(yieldExpressionAST);

        if (genContext.passResult)
        {
            genContext.passResult->functionReturnTypeShouldBeProvided = true;
        }

        // get state
        auto state = 0;
        if (genContext.state)
        {
            state = (*genContext.state)++;
        }
        else
        {
            assert(false);
        }

        // set restore point (return point)
        stringstream num;
        num << state;

        NodeFactory nf(NodeFactoryFlags::None);

        // save return point - state
        auto setStateExpr = nf.createBinaryExpression(
            nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword), nf.createIdentifier(S("step"))),
            nf.createToken(SyntaxKind::EqualsToken), nf.createNumericLiteral(num.str(), TokenFlags::None));

        mlirGen(setStateExpr, genContext);

        // return value
        auto yieldRetValue = getYieldReturnObject(nf, yieldExpressionAST->expression, false);
        auto yieldValue = mlirGen(yieldRetValue, genContext);

        mlirGenReturnValue(location, yieldValue, true, genContext);

        std::stringstream label;
        label << "state" << state;
        builder.create<mlir_ts::StateLabelOp>(location, label.str());

        // TODO: yield value to continue, should be loaded from "next(value)" parameter
        // return yieldValue;
        return mlir::Value();
    }

    mlir::Value mlirGen(AwaitExpression awaitExpressionAST, const GenContext &genContext)
    {
#ifdef ENABLE_ASYNC
        auto location = loc(awaitExpressionAST);

        auto resultType = evaluate(awaitExpressionAST->expression, genContext);

        auto asyncExecOp = builder.create<mlir::async::ExecuteOp>(
            location, resultType ? mlir::TypeRange{resultType} : mlir::TypeRange(), mlir::ValueRange{}, mlir::ValueRange{},
            [&](mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange values) {
                SmallVector<mlir::Type, 0> types;
                SmallVector<mlir::Value, 0> operands;
                if (resultType)
                {
                    types.push_back(resultType);
                }

                auto value = mlirGen(awaitExpressionAST->expression, genContext);
                if (value)
                {
                    builder.create<mlir::async::YieldOp>(location, mlir::ValueRange{value});
                }
                else
                {
                    builder.create<mlir::async::YieldOp>(location, mlir::ValueRange{});
                }
            });
        if (resultType)
        {
            auto asyncAwaitOp = builder.create<mlir::async::AwaitOp>(location, asyncExecOp.results().back());
            return asyncAwaitOp.getResult(0);
        }
        else
        {
            auto asyncAwaitOp = builder.create<mlir::async::AwaitOp>(location, asyncExecOp.token());
        }

        return mlir::Value();
#else
        return mlirGen(awaitExpressionAST->expression, genContext);
#endif
    }

    mlir::LogicalResult processReturnType(mlir::Value expressionValue, const GenContext &genContext)
    {
        // record return type if not provided
        if (genContext.passResult)
        {
            if (!expressionValue)
            {
                return mlir::failure();
            }

            auto type = expressionValue.getType();
            LLVM_DEBUG(dbgs() << "\n!! store return type: " << type << "\n\n");

            // if return type is not detected, take first and exit
            if (!genContext.passResult->functionReturnType)
            {
                genContext.passResult->functionReturnType = type;
                return mlir::success();
            }

            auto undefType = getUndefinedType();
            auto nullType = getNullType();
            auto undefPlaceHolderType = getUndefPlaceHolderType();

            std::function<bool(mlir::Type)> testType;
            testType = [&](mlir::Type type) {
                if (type == undefType || type == nullType || type == undefPlaceHolderType)
                {
                    return false;
                }

                if (auto optType = type.dyn_cast_or_null<mlir_ts::OptionalType>())
                {
                    return testType(optType.getElementType());
                }

                return true;
            };

            // filter out types, such as: undefined, objects with undefined values etc
            if (type == undefType || type == nullType)
            {
                return mlir::failure();
            }

            MLIRTypeHelper mth(builder.getContext());
            if (mth.hasUndefines(type))
            {
                return mlir::failure();
            }

            if (mth.hasUndefines(genContext.passResult->functionReturnType))
            {
                if (!mth.isCastableTypes(genContext.passResult->functionReturnType, type))
                {
                    return mlir::failure();
                }
            }
            else if (!mth.isCastableTypes(type, genContext.passResult->functionReturnType))
            {
                return mlir::failure();
            }

            // TODO: use "mth.findBaseType(leftExpressionValue.getType(), resultWhenFalseType);" to find base type for both types
            // + auto defaultUnionType = getUnionType(resultWhenTrueType, resultWhenFalseType);

            // we can save result type after joining two types
            genContext.passResult->functionReturnType = type;
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenReturnValue(mlir::Location location, mlir::Value expressionValue, bool yieldReturn,
                                           const GenContext &genContext)
    {
        if (genContext.passResult)
        {
            genContext.passResult->functionReturnTypeShouldBeProvided = true;
        }

        auto funcOp = const_cast<GenContext &>(genContext).funcOp;
        if (funcOp)
        {
            auto countResults = funcOp.getCallableResults().size();
            if (countResults > 0)
            {
                auto returnType = funcOp.getCallableResults().front();

                if (!expressionValue)
                {
                    if (!genContext.allowPartialResolve)
                    {
                        emitError(location) << "'return' must have value";
                        return mlir::failure();
                    }
                }
                else if (returnType != expressionValue.getType())
                {
                    auto castValue = cast(location, returnType, expressionValue, genContext);
                    expressionValue = castValue;
                }
            }
        }

        // record return type if not provided
        processReturnType(expressionValue, genContext);

        if (!expressionValue)
        {
            emitError(location) << "'return' must have value";
            builder.create<mlir_ts::ReturnOp>(location);
            return mlir::success();
        }

        auto retVarInfo = symbolTable.lookup(RETURN_VARIABLE_NAME);
        if (!retVarInfo.second)
        {
            if (genContext.allowPartialResolve)
            {
                return mlir::success();
            }

            emitError(location) << "can't find return variable";
            return mlir::failure();
        }

        if (yieldReturn)
        {
            builder.create<mlir_ts::YieldReturnValOp>(location, expressionValue, retVarInfo.first);
        }
        else
        {
            builder.create<mlir_ts::ReturnValOp>(location, expressionValue, retVarInfo.first);
        }

        return mlir::success();
    }

    mlir::LogicalResult addSafeCastStatement(Expression expr, Node typeToken, const GenContext &genContext)
    {
        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _safe_casted = expr;
        declarations.push_back(nf.createVariableDeclaration(_safe_casted, undefined, undefined, nf.createTypeAssertion(typeToken, expr)));

        auto varDeclList = nf.createVariableDeclarationList(declarations, NodeFlags::Const);

        auto expr_statement = nf.createVariableStatement(undefined, varDeclList);

        const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());

        return mlir::success();
    }

    mlir::LogicalResult checkSafeCastTypeOf(Expression typeOfVal, Expression constVal, const GenContext &genContext)
    {
        if (auto typeOfOp = typeOfVal.as<TypeOfExpression>())
        {
            // strip parenthesizes
            auto expr = stripParentheses(typeOfOp->expression);
            if (!expr.is<Identifier>())
            {
                return mlir::failure();
            }

            if (auto stringLiteral = constVal.as<ts::StringLiteral>())
            {
                // create 'expression' = <string>'expression;
                NodeFactory nf(NodeFactoryFlags::None);

                auto text = stringLiteral->text;
                Node typeToken;
                if (text == S("string"))
                {
                    typeToken = nf.createToken(SyntaxKind::StringKeyword);
                }
                else if (text == S("number"))
                {
                    typeToken = nf.createToken(SyntaxKind::NumberKeyword);
                }
                else if (text == S("boolean"))
                {
                    typeToken = nf.createToken(SyntaxKind::BooleanKeyword);
                }

                if (typeToken)
                {
                    addSafeCastStatement(expr, typeToken, genContext);
                }

                return mlir::success();
            }
        }

        return mlir::failure();
    }

    Expression stripParentheses(Expression exprVal)
    {
        auto expr = exprVal;
        while (expr.is<ParenthesizedExpression>())
        {
            expr = expr.as<ParenthesizedExpression>()->expression;
        }

        return expr;
    }

    mlir::LogicalResult checkSafeCastPropertyAccessLogic(TextRange textRange, Expression objAccessExpression, mlir::Type typeOfObject,
                                                         Node name, mlir::Value constVal, const GenContext &genContext)
    {
        if (auto unionType = typeOfObject.dyn_cast<mlir_ts::UnionType>())
        {
            auto isConst = false;
            mlir::Attribute value;
            isConst = isConstValue(constVal);
            if (isConst)
            {
                auto constantOp = constVal.getDefiningOp<mlir_ts::ConstantOp>();
                assert(constantOp);
                auto valueAttr = constantOp.valueAttr();

                MLIRCodeLogic mcl(builder);
                auto namePtr = MLIRHelper::getName(name, stringAllocator);
                assert(!namePtr.empty());
                auto fieldNameAttr = mcl.TupleFieldName(namePtr);

                for (auto unionSubType : unionType.getTypes())
                {
                    if (auto tupleType = unionSubType.dyn_cast<mlir_ts::TupleType>())
                    {
                        auto fieldIndex = tupleType.getIndex(fieldNameAttr);
                        auto fieldType = tupleType.getType(fieldIndex);
                        if (auto literalType = fieldType.dyn_cast<mlir_ts::LiteralType>())
                        {
                            if (literalType.getValue() == valueAttr)
                            {
                                // enable safe cast found
                                auto typeAliasNameUTF8 = MLIRHelper::getAnonymousName(loc_check(textRange), "ta_");
                                auto typeAliasName = ConvertUTF8toWide(typeAliasNameUTF8);
                                const_cast<GenContext &>(genContext).typeAliasMap.insert({typeAliasNameUTF8, tupleType});

                                NodeFactory nf(NodeFactoryFlags::None);
                                auto typeRef = nf.createTypeReferenceNode(nf.createIdentifier(typeAliasName));
                                addSafeCastStatement(objAccessExpression, typeRef, genContext);
                                break;
                            }
                        }
                    }
                }
            }
        }

        return mlir::failure();
    }

    mlir::LogicalResult checkSafeCastPropertyAccess(Expression exprVal, Expression constVal, const GenContext &genContext)
    {
        auto expr = stripParentheses(exprVal);
        if (expr.is<PropertyAccessExpression>())
        {
            auto propertyAccessExpressionOp = expr.as<PropertyAccessExpression>();
            auto objAccessExpression = propertyAccessExpressionOp->expression;
            auto typeOfObject = evaluate(objAccessExpression, genContext);

            LLVM_DEBUG(llvm::dbgs() << "\n!! SafeCastCheck: " << typeOfObject << "");

            evaluate(
                constVal,
                [&](mlir::Value val) {
                    checkSafeCastPropertyAccessLogic(constVal, objAccessExpression, typeOfObject, propertyAccessExpressionOp->name, val,
                                                     genContext);
                },
                genContext);
        }

        return mlir::failure();
    }

    mlir::LogicalResult checkSafeCast(Expression expr, const GenContext &genContext)
    {
        if (expr != SyntaxKind::BinaryExpression)
        {
            return mlir::success();
        }

        if (auto binExpr = expr.as<BinaryExpression>())
        {
            auto op = (SyntaxKind)binExpr->operatorToken;
            if (op == SyntaxKind::EqualsEqualsToken || op == SyntaxKind::EqualsEqualsEqualsToken)
            {
                auto left = binExpr->left;
                auto right = binExpr->right;

                if (mlir::failed(checkSafeCastTypeOf(left, right, genContext)))
                {
                    if (mlir::failed(checkSafeCastTypeOf(right, left, genContext)))
                    {
                        if (mlir::failed(checkSafeCastPropertyAccess(left, right, genContext)))
                        {
                            return checkSafeCastPropertyAccess(right, left, genContext);
                        }
                    }
                }

                return mlir::success();
            }

            if (op == SyntaxKind::InstanceOfKeyword)
            {
                auto instanceOf = binExpr;
                if (instanceOf->left.is<Identifier>())
                {
                    NodeFactory nf(NodeFactoryFlags::None);
                    addSafeCastStatement(instanceOf->left, nf.createTypeReferenceNode(instanceOf->right), genContext);
                    return mlir::success();
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(IfStatement ifStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(ifStatementAST);

        auto hasElse = !!ifStatementAST->elseStatement;

        // condition
        auto condValue = mlirGen(ifStatementAST->expression, genContext);

        VALIDATE_LOGIC(condValue, location)

        if (condValue.getType() != getBooleanType())
        {
            condValue = cast(location, getBooleanType(), condValue, genContext);
        }

        auto ifOp = builder.create<mlir_ts::IfOp>(location, condValue, hasElse);

        // check if we do safe-cast here
        checkSafeCast(ifStatementAST->expression, genContext);

        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        mlirGen(ifStatementAST->thenStatement, genContext);

        if (hasElse)
        {
            builder.setInsertionPointToStart(&ifOp.elseRegion().front());
            mlirGen(ifStatementAST->elseStatement, genContext);
        }

        builder.setInsertionPointAfter(ifOp);

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(DoStatement doStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(doStatementAST);

        SmallVector<mlir::Type, 0> types;
        SmallVector<mlir::Value, 0> operands;

        auto doWhileOp = builder.create<mlir_ts::DoWhileOp>(location, types, operands);
        if (!label.empty())
        {
            doWhileOp->setAttr(LABEL_ATTR_NAME, builder.getStringAttr(label));
            label = "";
        }

        /*auto *cond =*/builder.createBlock(&doWhileOp.cond(), {}, types);
        /*auto *body =*/builder.createBlock(&doWhileOp.body(), {}, types);

        // body in condition
        builder.setInsertionPointToStart(&doWhileOp.body().front());
        mlirGen(doStatementAST->statement, genContext);
        // just simple return, as body in cond
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointToStart(&doWhileOp.cond().front());
        auto conditionValue = mlirGen(doStatementAST->expression, genContext);

        VALIDATE_LOGIC(conditionValue, location)

        if (conditionValue.getType() != getBooleanType())
        {
            conditionValue = cast(location, getBooleanType(), conditionValue, genContext);
        }

        builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});

        builder.setInsertionPointAfter(doWhileOp);
        return mlir::success();
    }

    mlir::LogicalResult mlirGen(WhileStatement whileStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(whileStatementAST);

        SmallVector<mlir::Type, 0> types;
        SmallVector<mlir::Value, 0> operands;

        auto whileOp = builder.create<mlir_ts::WhileOp>(location, types, operands);
        if (!label.empty())
        {
            whileOp->setAttr(LABEL_ATTR_NAME, builder.getStringAttr(label));
            label = "";
        }

        /*auto *cond =*/builder.createBlock(&whileOp.cond(), {}, types);
        /*auto *body =*/builder.createBlock(&whileOp.body(), {}, types);

        // condition
        builder.setInsertionPointToStart(&whileOp.cond().front());
        auto conditionValue = mlirGen(whileStatementAST->expression, genContext);

        VALIDATE_LOGIC(conditionValue, location)

        if (conditionValue.getType() != getBooleanType())
        {
            conditionValue = cast(location, getBooleanType(), conditionValue, genContext);
        }

        builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});

        // body
        builder.setInsertionPointToStart(&whileOp.body().front());
        mlirGen(whileStatementAST->statement, genContext);
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(whileOp);
        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ForStatement forStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forStatementAST);

        auto hasAwait = TransformFlags::ForAwait == (forStatementAST->transformFlags & TransformFlags::ForAwait);

        // initializer
        // TODO: why do we have ForInitialier
        if (forStatementAST->initializer.is<Expression>())
        {
            auto init = mlirGen(forStatementAST->initializer.as<Expression>(), genContext);
            if (!init)
            {
                return mlir::failure();
            }
        }
        else if (forStatementAST->initializer.is<VariableDeclarationList>())
        {
            auto result = mlirGen(forStatementAST->initializer.as<VariableDeclarationList>(), genContext);
            if (failed(result))
            {
                return result;
            }
        }

        SmallVector<mlir::Type, 0> types;
        SmallVector<mlir::Value, 0> operands;

        mlir::Value asyncGroupResult;
        if (hasAwait)
        {
            auto groupType = mlir::async::GroupType::get(builder.getContext());
            auto blockSize = builder.create<mlir_ts::ConstantOp>(location, builder.getIndexAttr(0));
            auto asyncGroupOp = builder.create<mlir::async::CreateGroupOp>(location, groupType, blockSize);
            asyncGroupResult = asyncGroupOp.result();
            // operands.push_back(asyncGroupOp);
            // types.push_back(groupType);
        }

        auto forOp = builder.create<mlir_ts::ForOp>(location, types, operands);
        if (!label.empty())
        {
            forOp->setAttr(LABEL_ATTR_NAME, builder.getStringAttr(label));
            label = "";
        }

        /*auto *cond =*/builder.createBlock(&forOp.cond(), {}, types);
        /*auto *body =*/builder.createBlock(&forOp.body(), {}, types);
        /*auto *incr =*/builder.createBlock(&forOp.incr(), {}, types);

        builder.setInsertionPointToStart(&forOp.cond().front());
        auto conditionValue = mlirGen(forStatementAST->condition, genContext);
        if (conditionValue)
        {
            builder.create<mlir_ts::ConditionOp>(location, conditionValue, mlir::ValueRange{});
        }
        else
        {
            builder.create<mlir_ts::NoConditionOp>(location, mlir::ValueRange{});
        }

        // body
        builder.setInsertionPointToStart(&forOp.body().front());
        if (hasAwait)
        {
            if (forStatementAST->statement == SyntaxKind::Block)
            {
                // TODO: it is kind of hack, maybe you can find better solution
                auto firstStatement = forStatementAST->statement.as<Block>()->statements.front();
                mlirGen(firstStatement, genContext);
                firstStatement->processed = true;
            }

            // async body
            auto asyncExecOp =
                builder.create<mlir::async::ExecuteOp>(location, mlir::TypeRange{}, mlir::ValueRange{}, mlir::ValueRange{},
                                                       [&](mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange values) {
                                                           GenContext execOpBodyGenContext(genContext);
                                                           execOpBodyGenContext.skipProcessed = true;
                                                           mlirGen(forStatementAST->statement, execOpBodyGenContext);
                                                           builder.create<mlir::async::YieldOp>(location, mlir::ValueRange{});
                                                       });

            // add to group
            auto rankType = mlir::IndexType::get(builder.getContext());
            // TODO: should i replace with value from arg0?
            builder.create<mlir::async::AddToGroupOp>(location, rankType, asyncExecOp.token(), asyncGroupResult);
        }
        else
        {
            // default
            mlirGen(forStatementAST->statement, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location);

        // increment
        builder.setInsertionPointToStart(&forOp.incr().front());
        mlirGen(forStatementAST->incrementor, genContext);
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(forOp);

        if (hasAwait)
        {
            // Not helping
            /*
            // async await all, see convert-to-llvm.mlir
            auto asyncExecAwaitAllOp =
                builder.create<mlir::async::ExecuteOp>(location, mlir::TypeRange{}, mlir::ValueRange{}, mlir::ValueRange{},
                                                       [&](mlir::OpBuilder &builder, mlir::Location location, mlir::ValueRange values) {
                                                           builder.create<mlir::async::AwaitAllOp>(location, asyncGroupResult);
                                                           builder.create<mlir::async::YieldOp>(location, mlir::ValueRange{});
                                                       });
            */

            // Wait for the completion of all subtasks.
            builder.create<mlir::async::AwaitAllOp>(location, asyncGroupResult);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ForInStatement forInStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forInStatementAST);

        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _i = nf.createIdentifier(S("_i_"));
        declarations.push_back(nf.createVariableDeclaration(_i, undefined, undefined, nf.createNumericLiteral(S("0"))));

        auto _a = nf.createIdentifier(S("_a_"));
        auto arrayVar = nf.createVariableDeclaration(_a, undefined, undefined, forInStatementAST->expression);
        arrayVar->transformFlags |= TransformFlags::ForceConstRef;
        declarations.push_back(arrayVar);

        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let);

        // condition
        // auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
        // nf.createCallExpression(nf.createIdentifier(S("#_last_field")), undefined, NodeArray<Expression>(_a)));
        auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
                                              nf.createPropertyAccessExpression(_a, nf.createIdentifier(S("length"))));

        // incr
        auto incr = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::PlusPlusToken), _i);

        // block
        NodeArray<ts::Statement> statements;

        auto varDeclList = forInStatementAST->initializer.as<VariableDeclarationList>();
        varDeclList->declarations.front()->initializer = _i;

        statements.push_back(nf.createVariableStatement(undefined, varDeclList));
        statements.push_back(forInStatementAST->statement);
        auto block = nf.createBlock(statements);

        // final For statement
        auto forStatNode = nf.createForStatement(initVars, cond, incr, block);

        return mlirGen(forStatNode, genContext);
    }

    mlir::LogicalResult mlirGenES3(ForOfStatement forOfStatementAST, mlir::Value exprValue, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forOfStatementAST);

        auto varDecl = std::make_shared<VariableDeclarationDOM>(EXPR_TEMPVAR_NAME, exprValue.getType(), location);
        declare(varDecl, exprValue, genContext);

        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _i = nf.createIdentifier(S("_i_"));
        declarations.push_back(nf.createVariableDeclaration(_i, undefined, undefined, nf.createNumericLiteral(S("0"))));

        auto _a = nf.createIdentifier(S("_a_"));
        auto arrayVar = nf.createVariableDeclaration(_a, undefined, undefined, nf.createIdentifier(S(EXPR_TEMPVAR_NAME)));
        arrayVar->transformFlags |= TransformFlags::ForceConstRef;

        declarations.push_back(arrayVar);

        // condition
        auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
                                              nf.createPropertyAccessExpression(_a, nf.createIdentifier(S("length"))));

        // incr
        auto incr = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::PlusPlusToken), _i);

        // block
        NodeArray<ts::Statement> statements;

        NodeArray<VariableDeclaration> varOfConstDeclarations;
        auto _ci = nf.createIdentifier(S("_ci_"));
        varOfConstDeclarations.push_back(nf.createVariableDeclaration(_ci, undefined, undefined, _i));
        auto varsOfConst = nf.createVariableDeclarationList(varOfConstDeclarations, NodeFlags::Const);

        auto varDeclList = forOfStatementAST->initializer.as<VariableDeclarationList>();
        varDeclList->declarations.front()->initializer = nf.createElementAccessExpression(_a, _ci);

        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let /*varDeclList->flags*/);

        // in async exec, we will put first statement outside fo async.exec, to convert ref<int> into <int>
        statements.push_back(nf.createVariableStatement(undefined, varsOfConst));
        statements.push_back(nf.createVariableStatement(undefined, varDeclList));
        statements.push_back(forOfStatementAST->statement);
        auto block = nf.createBlock(statements);

        // final For statement
        auto forStatNode = nf.createForStatement(initVars, cond, incr, block);
        if (forOfStatementAST->awaitModifier)
        {
            forStatNode->transformFlags |= TransformFlags::ForAwait;
        }

        return mlirGen(forStatNode, genContext);
    }

    mlir::LogicalResult mlirGenES2015(ForOfStatement forOfStatementAST, mlir::Value exprValue, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forOfStatementAST);

        auto varDecl = std::make_shared<VariableDeclarationDOM>(EXPR_TEMPVAR_NAME, exprValue.getType(), location);
        declare(varDecl, exprValue, genContext);

        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _b = nf.createIdentifier(S("_b_"));
        auto _next = nf.createIdentifier(S("next"));
        auto _bVar = nf.createVariableDeclaration(_b, undefined, undefined, nf.createIdentifier(S(EXPR_TEMPVAR_NAME)));
        declarations.push_back(_bVar);

        NodeArray<Expression> nextArgs;

        auto _c = nf.createIdentifier(S("_c_"));
        auto _done = nf.createIdentifier(S("done"));
        auto _value = nf.createIdentifier(S("value"));
        auto _cVar = nf.createVariableDeclaration(
            _c, undefined, undefined, nf.createCallExpression(nf.createPropertyAccessExpression(_b, _next), undefined, nextArgs));
        declarations.push_back(_cVar);

        // condition
        auto cond =
            nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::ExclamationToken), nf.createPropertyAccessExpression(_c, _done));

        // incr
        auto incr = nf.createBinaryExpression(_c, nf.createToken(SyntaxKind::EqualsToken),
                                              nf.createCallExpression(nf.createPropertyAccessExpression(_b, _next), undefined, nextArgs));

        // block
        NodeArray<ts::Statement> statements;

        auto varDeclList = forOfStatementAST->initializer.as<VariableDeclarationList>();
        varDeclList->declarations.front()->initializer = nf.createPropertyAccessExpression(_c, _value);

        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let /*varDeclList->flags*/);

        statements.push_back(nf.createVariableStatement(undefined, varDeclList));
        statements.push_back(forOfStatementAST->statement);
        auto block = nf.createBlock(statements);

        // final For statement
        auto forStatNode = nf.createForStatement(initVars, cond, incr, block);
        if (forOfStatementAST->awaitModifier)
        {
            forStatNode->transformFlags |= TransformFlags::ForAwait;
        }

        return mlirGen(forStatNode, genContext);
    }

    mlir::LogicalResult mlirGen(ForOfStatement forOfStatementAST, const GenContext &genContext)
    {
        auto location = loc(forOfStatementAST);

        auto exprValue = mlirGen(forOfStatementAST->expression, genContext);

        auto propertyType = evaluateProperty(exprValue, "next", genContext);
        if (propertyType)
        {
            if (mlir::succeeded(mlirGenES2015(forOfStatementAST, exprValue, genContext)))
            {
                return mlir::success();
            }
        }

        return mlirGenES3(forOfStatementAST, exprValue, genContext);
    }

    mlir::LogicalResult mlirGen(LabeledStatement labeledStatementAST, const GenContext &genContext)
    {
        auto location = loc(labeledStatementAST);

        label = MLIRHelper::getName(labeledStatementAST->label);

        auto kind = (SyntaxKind)labeledStatementAST->statement;
        if (kind == SyntaxKind::EmptyStatement && StringRef(label).startswith("state"))
        {
            builder.create<mlir_ts::StateLabelOp>(location, builder.getStringAttr(label));
            return mlir::success();
        }

        auto noLabelOp = kind == SyntaxKind::WhileStatement || kind == SyntaxKind::DoStatement || kind == SyntaxKind::ForStatement ||
                         kind == SyntaxKind::ForInStatement || kind == SyntaxKind::ForOfStatement;

        if (noLabelOp)
        {
            auto res = mlirGen(labeledStatementAST->statement, genContext);
            return res;
        }

        auto labelOp = builder.create<mlir_ts::LabelOp>(location, builder.getStringAttr(label));

        // add merge block
        labelOp.addMergeBlock();
        auto *mergeBlock = labelOp.getMergeBlock();

        builder.setInsertionPointToStart(mergeBlock);

        auto res = mlirGen(labeledStatementAST->statement, genContext);

        builder.setInsertionPointAfter(labelOp);

        return res;
    }

    mlir::LogicalResult mlirGen(DebuggerStatement debuggerStatementAST, const GenContext &genContext)
    {
        auto location = loc(debuggerStatementAST);

        builder.create<mlir_ts::DebuggerOp>(location);
        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ContinueStatement continueStatementAST, const GenContext &genContext)
    {
        auto location = loc(continueStatementAST);

        auto label = MLIRHelper::getName(continueStatementAST->label);

        builder.create<mlir_ts::ContinueOp>(location, builder.getStringAttr(label));
        return mlir::success();
    }

    mlir::LogicalResult mlirGen(BreakStatement breakStatementAST, const GenContext &genContext)
    {
        auto location = loc(breakStatementAST);

        auto label = MLIRHelper::getName(breakStatementAST->label);

        builder.create<mlir_ts::BreakOp>(location, builder.getStringAttr(label));
        return mlir::success();
    }

    mlir::LogicalResult mlirGenSwitchCase(mlir::Location location, Expression switchExpr, mlir::Value switchValue,
                                          NodeArray<ts::CaseOrDefaultClause> &clauses, int index, mlir::Block *mergeBlock,
                                          mlir::Block *&defaultBlock, SmallVector<mlir::CondBranchOp> &pendingConditions,
                                          SmallVector<mlir::BranchOp> &pendingBranches, mlir::Operation *&previousConditionOrFirstBranchOp,
                                          std::function<void(Expression, mlir::Value)> extraCode, const GenContext &genContext)
    {
        SymbolTableScopeT safeCastVarScope(symbolTable);

        enum
        {
            trueIndex = 0,
            falseIndex = 1
        };

        auto caseBlock = clauses[index];
        auto statements = caseBlock->statements;
        // inline block
        // TODO: should I inline block as it is isolator of local vars?
        if (statements.size() == 1)
        {
            auto firstStatement = statements.front();
            if ((SyntaxKind)firstStatement == SyntaxKind::Block)
            {
                statements = statements.front().as<Block>()->statements;
            }
        }

        auto setPreviousCondOrJumpOp = [&](mlir::Operation *jump, mlir::Block *where) {
            if (auto condOp = dyn_cast_or_null<mlir::CondBranchOp>(jump))
            {
                condOp->setSuccessor(where, falseIndex);
                return;
            }

            if (auto branchOp = dyn_cast_or_null<mlir::BranchOp>(jump))
            {
                branchOp.setDest(where);
                return;
            }

            llvm_unreachable("not implemented");
        };

        // condition
        auto isDefaultCase = SyntaxKind::DefaultClause == (SyntaxKind)caseBlock;
        auto isDefaultAsFirstCase = index == 0 && clauses.size() > 1;
        if (SyntaxKind::CaseClause == (SyntaxKind)caseBlock)
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            auto caseConditionBlock = builder.createBlock(mergeBlock);
            if (previousConditionOrFirstBranchOp)
            {
                setPreviousCondOrJumpOp(previousConditionOrFirstBranchOp, caseConditionBlock);
            }

            auto caseExpr = caseBlock.as<CaseClause>()->expression;
            auto caseValue = mlirGen(caseExpr, genContext);

            extraCode(caseExpr, caseValue);

            auto switchValueEffective = switchValue;
            if (switchValue.getType() != caseValue.getType())
            {
                switchValueEffective = cast(location, caseValue.getType(), switchValue, genContext);
            }

            auto condition = builder.create<mlir_ts::LogicalBinaryOp>(
                location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsToken), switchValueEffective, caseValue);

            auto conditionI1 = cast(location, builder.getI1Type(), condition, genContext);

            auto condBranchOp = builder.create<mlir::CondBranchOp>(location, conditionI1, mergeBlock, /*trueArguments=*/mlir::ValueRange{},
                                                                   defaultBlock ? defaultBlock : mergeBlock,
                                                                   /*falseArguments=*/mlir::ValueRange{});

            previousConditionOrFirstBranchOp = condBranchOp;

            pendingConditions.push_back(condBranchOp);
        }
        else if (isDefaultAsFirstCase)
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            /*auto defaultCaseJumpBlock =*/builder.createBlock(mergeBlock);

            // this is first default and there is more conditions
            // add jump to first condition
            auto branchOp = builder.create<mlir::BranchOp>(location, mergeBlock);

            previousConditionOrFirstBranchOp = branchOp;
        }

        // statements block
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            auto caseBodyBlock = builder.createBlock(mergeBlock);
            if (isDefaultCase)
            {
                defaultBlock = caseBodyBlock;
                if (!isDefaultAsFirstCase && previousConditionOrFirstBranchOp)
                {
                    setPreviousCondOrJumpOp(previousConditionOrFirstBranchOp, caseBodyBlock);
                }
            }

            // set pending BranchOps
            for (auto pendingBranch : pendingBranches)
            {
                pendingBranch.setDest(caseBodyBlock);
            }

            pendingBranches.clear();

            for (auto pendingCondition : pendingConditions)
            {
                pendingCondition.setSuccessor(caseBodyBlock, trueIndex);
            }

            pendingConditions.clear();

            // process body case
            if (genContext.generatedStatements.size() > 0)
            {
                // auto generated code
                for (auto &statement : genContext.generatedStatements)
                {
                    if (failed(mlirGen(statement, genContext)))
                    {
                        return mlir::failure();
                    }
                }

                // clean up
                const_cast<GenContext &>(genContext).generatedStatements.clear();
            }

            auto hasBreak = false;
            for (auto statement : statements)
            {
                if ((SyntaxKind)statement == SyntaxKind::BreakStatement)
                {
                    hasBreak = true;
                    break;
                }

                if (failed(mlirGen(statement, genContext)))
                {
                    return mlir::failure();
                }
            }

            // exit;
            auto branchOp = builder.create<mlir::BranchOp>(location, mergeBlock);
            if (!hasBreak && !isDefaultCase)
            {
                pendingBranches.push_back(branchOp);
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(SwitchStatement switchStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(switchStatementAST);

        auto switchExpr = switchStatementAST->expression;
        auto switchValue = mlirGen(switchExpr, genContext);

        VALIDATE_LOGIC(switchValue, location)

        auto switchOp = builder.create<mlir_ts::SwitchOp>(location, switchValue);

        GenContext switchGenContext(genContext);
        switchGenContext.allocateVarsOutsideOfOperation = true;
        switchGenContext.currentOperation = switchOp;
        switchGenContext.insertIntoParentScope = true;

        // add merge block
        switchOp.addMergeBlock();
        auto *mergeBlock = switchOp.getMergeBlock();

        auto &clauses = switchStatementAST->caseBlock->clauses;

        SmallVector<mlir::CondBranchOp> pendingConditions;
        SmallVector<mlir::BranchOp> pendingBranches;
        mlir::Operation *previousConditionOrFirstBranchOp = nullptr;
        mlir::Block *defaultBlock = nullptr;

        // to support safe cast
        std::function<void(Expression, mlir::Value)> safeCastLogic;
        if (switchExpr.is<PropertyAccessExpression>())
        {
            auto propertyAccessExpressionOp = switchExpr.as<PropertyAccessExpression>();
            auto objAccessExpression = propertyAccessExpressionOp->expression;
            auto typeOfObject = evaluate(objAccessExpression, switchGenContext);
            auto name = propertyAccessExpressionOp->name;

            safeCastLogic = [=, &switchGenContext](Expression caseExpr, mlir::Value constVal) {
                GenContext safeCastGenContext(switchGenContext);
                switchGenContext.insertIntoParentScope = false;

                // Safe Cast
                if (mlir::failed(checkSafeCastTypeOf(switchExpr, caseExpr, switchGenContext)))
                {
                    checkSafeCastPropertyAccessLogic(caseExpr, objAccessExpression, typeOfObject, name, constVal, switchGenContext);
                }
            };
        }
        else
        {
            safeCastLogic = [&](Expression caseExpr, mlir::Value constVal) {};
        }

        // process without default
        for (int index = 0; index < clauses.size(); index++)
        {
            if (mlir::failed(mlirGenSwitchCase(location, switchExpr, switchValue, clauses, index, mergeBlock, defaultBlock,
                                               pendingConditions, pendingBranches, previousConditionOrFirstBranchOp, safeCastLogic,
                                               switchGenContext)))
            {
                return mlir::failure();
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! SWITCH: " << switchOp << "\n");

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ThrowStatement throwStatementAST, const GenContext &genContext)
    {
        auto location = loc(throwStatementAST);

        auto exception = mlirGen(throwStatementAST->expression, genContext);

        auto throwOp = builder.create<mlir_ts::ThrowOp>(location, exception);

        if (!genContext.allowPartialResolve)
        {
#ifdef WIN_EXCEPTION
            MLIRRTTIHelperVCWin32 rtti(builder, theModule);
#else
            MLIRRTTIHelperVCLinux rtti(builder, theModule);
#endif
            rtti.setRTTIForType(location, exception.getType(),
                                [&](StringRef classFullName) { return getClassInfoByFullName(classFullName); });
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(TryStatement tryStatementAST, const GenContext &genContext)
    {
        auto location = loc(tryStatementAST);

        std::string varName;
        auto catchClause = tryStatementAST->catchClause;
        if (catchClause)
        {
            auto varDecl = catchClause->variableDeclaration;
            if (varDecl)
            {
                varName = MLIRHelper::getName(varDecl->name);
                if (mlir::failed(mlirGen(varDecl, VariableClass::Let, genContext)))
                {
                    return mlir::failure();
                }
            }
        }

        const_cast<GenContext &>(genContext).funcOp.personalityAttr(builder.getBoolAttr(true));

        auto tryOp = builder.create<mlir_ts::TryOp>(location);
        /*
        tryOp->setAttr("try_id", builder.getI64IntegerAttr((int64_t)tryOp.getOperation()));

        auto parentTryOp = tryOp->getParentOfType<mlir_ts::TryOp>();
        if (parentTryOp)
        {
            tryOp->setAttr("unwind_to", builder.getI64IntegerAttr((int64_t)parentTryOp.getOperation()));
        }
        */

        GenContext tryGenContext(genContext);
        tryGenContext.allocateVarsOutsideOfOperation = true;
        tryGenContext.currentOperation = tryOp;

        SmallVector<mlir::Type, 0> types;

        /*auto *body =*/builder.createBlock(&tryOp.body(), {}, types);
        /*auto *catches =*/builder.createBlock(&tryOp.catches(), {}, types);
        /*auto *finallyBlock =*/builder.createBlock(&tryOp.finallyBlock(), {}, types);

        // body
        builder.setInsertionPointToStart(&tryOp.body().front());
        auto result = mlirGen(tryStatementAST->tryBlock, tryGenContext);
        if (mlir::failed(result))
        {
            return mlir::failure();
        }

        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        // catches
        builder.setInsertionPointToStart(&tryOp.catches().front());
        if (catchClause && catchClause->block)
        {
            if (!varName.empty())
            {
                MLIRCodeLogic mcl(builder);
                auto varInfo = resolveIdentifier(location, varName, tryGenContext);
                auto varRef = mcl.GetReferenceOfLoadOp(varInfo);
                builder.create<mlir_ts::CatchOp>(location, varRef);

                if (!genContext.allowPartialResolve)
                {
#ifdef WIN_EXCEPTION
                    MLIRRTTIHelperVCWin32 rtti(builder, theModule);
#else
                    MLIRRTTIHelperVCLinux rtti(builder, theModule);
#endif
                    rtti.setRTTIForType(location, varInfo.getType(),
                                        [&](StringRef classFullName) { return getClassInfoByFullName(classFullName); });
                }
            }

            result = mlirGen(tryStatementAST->catchClause->block, tryGenContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }
        }

        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        // finally
        builder.setInsertionPointToStart(&tryOp.finallyBlock().front());
        if (tryStatementAST->finallyBlock)
        {
            result = mlirGen(tryStatementAST->finallyBlock, tryGenContext);
            if (mlir::failed(result))
            {
                return mlir::failure();
            }
        }

        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(tryOp);
        return result;
    }

    mlir::Value mlirGen(UnaryExpression unaryExpressionAST, const GenContext &genContext)
    {
        return mlirGen(unaryExpressionAST.as<Expression>(), genContext);
    }

    mlir::Value mlirGen(LeftHandSideExpression leftHandSideExpressionAST, const GenContext &genContext)
    {
        return mlirGen(leftHandSideExpressionAST.as<Expression>(), genContext);
    }

    mlir::Value mlirGen(PrefixUnaryExpression prefixUnaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(prefixUnaryExpressionAST);

        auto opCode = prefixUnaryExpressionAST->_operator;

        auto expression = prefixUnaryExpressionAST->operand;
        auto expressionValue = mlirGen(expression, genContext);

        VALIDATE(expressionValue, location)

        auto boolValue = expressionValue;

        switch (opCode)
        {
        case SyntaxKind::ExclamationToken:

            if (expressionValue.getType() != getBooleanType())
            {
                boolValue = cast(location, getBooleanType(), expressionValue, genContext);
            }

            return builder.create<mlir_ts::ArithmeticUnaryOp>(location, getBooleanType(), builder.getI32IntegerAttr((int)opCode),
                                                              boolValue);
        case SyntaxKind::TildeToken:
        case SyntaxKind::PlusToken:
        case SyntaxKind::MinusToken:
            return builder.create<mlir_ts::ArithmeticUnaryOp>(location, expressionValue.getType(), builder.getI32IntegerAttr((int)opCode),
                                                              expressionValue);
        case SyntaxKind::PlusPlusToken:
        case SyntaxKind::MinusMinusToken:
            return builder.create<mlir_ts::PrefixUnaryOp>(location, expressionValue.getType(), builder.getI32IntegerAttr((int)opCode),
                                                          expressionValue);
        default:
            llvm_unreachable("not implemented");
        }
    }

    mlir::Value mlirGen(PostfixUnaryExpression postfixUnaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(postfixUnaryExpressionAST);

        auto opCode = postfixUnaryExpressionAST->_operator;

        auto expression = postfixUnaryExpressionAST->operand;
        auto expressionValue = mlirGen(expression, genContext);

        VALIDATE(expressionValue, location)

        switch (opCode)
        {
        case SyntaxKind::PlusPlusToken:
        case SyntaxKind::MinusMinusToken:
            return builder.create<mlir_ts::PostfixUnaryOp>(location, expressionValue.getType(), builder.getI32IntegerAttr((int)opCode),
                                                           expressionValue);
        default:
            llvm_unreachable("not implemented");
        }
    }

    mlir::Value mlirGen(ConditionalExpression conditionalExpressionAST, const GenContext &genContext)
    {
        auto location = loc(conditionalExpressionAST);

        // condition
        auto condExpression = conditionalExpressionAST->condition;
        auto condValue = mlirGen(condExpression, genContext);

        VALIDATE(condValue, location);

        if (condValue.getType() != getBooleanType())
        {
            condValue = cast(location, getBooleanType(), condValue, genContext);
        }

        // detect value type
        // TODO: sync types for 'when' and 'else'
        MLIRTypeHelper mth(builder.getContext());
        auto resultWhenTrueType = evaluate(conditionalExpressionAST->whenTrue, genContext);
        auto resultWhenFalseType = evaluate(conditionalExpressionAST->whenFalse, genContext);
        auto defaultUnionType = getUnionType(resultWhenTrueType, resultWhenFalseType);
        auto resultType = mth.findBaseType(resultWhenTrueType, resultWhenFalseType, defaultUnionType);

        if (genContext.allowPartialResolve)
        {
            if (!resultType)
            {
                return mlir::Value();
            }

            if (!resultWhenTrueType || !resultWhenFalseType)
            {
                // return undef value
            }

            auto udef = builder.create<mlir_ts::UndefOp>(location, mlir::TypeRange{resultType});
            return udef;
        }

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, condValue, true);

        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        auto whenTrueExpression = conditionalExpressionAST->whenTrue;
        auto resultTrue = mlirGen(whenTrueExpression, genContext);

        VALIDATE(resultTrue, location);

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{cast(location, resultType, resultTrue, genContext)});

        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
        auto whenFalseExpression = conditionalExpressionAST->whenFalse;
        auto resultFalse = mlirGen(whenFalseExpression, genContext);

        VALIDATE(resultFalse, location);

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{cast(location, resultType, resultFalse, genContext)});

        builder.setInsertionPointAfter(ifOp);

        return ifOp.getResult(0);
    }

    mlir::Value mlirGenAndOrLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext, bool andOp)
    {
        auto location = loc(binaryExpressionAST);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        // condition
        auto leftExpressionValue = mlirGen(leftExpression, genContext);

        VALIDATE(leftExpressionValue, location)

        MLIRTypeHelper mth(builder.getContext());
        auto resultWhenFalseType = evaluate(rightExpression, genContext);
        auto defaultUnionType = getUnionType(leftExpressionValue.getType(), resultWhenFalseType);
        auto resultType = andOp ? mth.findBaseType(resultWhenFalseType, leftExpressionValue.getType(), defaultUnionType)
                                : mth.findBaseType(leftExpressionValue.getType(), resultWhenFalseType, defaultUnionType);

        auto condValue = cast(location, getBooleanType(), leftExpressionValue, genContext);

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, condValue, true);

        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        auto resultTrue = andOp ? mlirGen(rightExpression, genContext) : leftExpressionValue;

        if (andOp)
        {
            VALIDATE(resultTrue, location)
        }

        // sync left part
        if (resultType != resultTrue.getType())
        {
            resultTrue = cast(location, resultType, resultTrue, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultTrue});

        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
        auto resultFalse = andOp ? leftExpressionValue : mlirGen(rightExpression, genContext);

        if (!andOp)
        {
            VALIDATE(resultFalse, location)
        }

        // sync right part
        if (resultType != resultFalse.getType())
        {
            resultFalse = cast(location, resultType, resultFalse, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultFalse});

        builder.setInsertionPointAfter(ifOp);

        return ifOp.results().front();
    }

    mlir::Value mlirGenQuestionQuestionLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        // condition
        auto leftExpressionValue = mlirGen(leftExpression, genContext);

        VALIDATE(leftExpressionValue, location)

        MLIRTypeHelper mth(builder.getContext());
        auto resultWhenFalseType = evaluate(rightExpression, genContext);
        auto defaultUnionType = getUnionType(leftExpressionValue.getType(), resultWhenFalseType);
        auto resultType = mth.findBaseType(resultWhenFalseType, leftExpressionValue.getType(), defaultUnionType);

        auto methodPtr = cast(location, getOpaqueType(), leftExpressionValue, genContext);

        auto nullVal = builder.create<mlir_ts::NullOp>(location, getNullType());
        auto compareToNull = builder.create<mlir_ts::LogicalBinaryOp>(
            location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsToken), methodPtr, nullVal);

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, compareToNull, true);

        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        auto resultTrue = mlirGen(rightExpression, genContext);

        VALIDATE(resultTrue, location)

        // sync left part
        if (resultType != resultTrue.getType())
        {
            resultTrue = cast(location, resultType, resultTrue, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultTrue});

        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
        auto resultFalse = leftExpressionValue;

        // sync right part
        if (resultType != resultFalse.getType())
        {
            resultFalse = cast(location, resultType, resultFalse, genContext);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultFalse});

        builder.setInsertionPointAfter(ifOp);

        return ifOp.results().front();
    }

    mlir::Value mlirGenInLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        // Supports only array now
        auto location = loc(binaryExpressionAST);

        NodeFactory nf(NodeFactoryFlags::None);

        // condition
        auto cond =
            nf.createBinaryExpression(binaryExpressionAST->left, nf.createToken(SyntaxKind::LessThanToken),
                                      nf.createPropertyAccessExpression(binaryExpressionAST->right, nf.createIdentifier(S("length"))));

        return mlirGen(cond, genContext);
    }

    mlir::Value mlirGenCallThisMethod(mlir::Location location, mlir::Value thisValue, StringRef methodName,
                                      NodeArray<TypeNode> typeArguments, NodeArray<Expression> arguments, const GenContext &genContext)
    {
        // to remove temp var .ctor after call
        SymbolTableScopeT varScope(symbolTable);

        auto varDecl = std::make_shared<VariableDeclarationDOM>(THIS_TEMPVAR_NAME, thisValue.getType(), location);
        declare(varDecl, thisValue, genContext);

        NodeFactory nf(NodeFactoryFlags::None);

        auto thisToken = nf.createIdentifier(S(THIS_TEMPVAR_NAME));
        auto callLogic = nf.createCallExpression(nf.createPropertyAccessExpression(thisToken, nf.createIdentifier(stows(methodName.str()))),
                                                 typeArguments, arguments);

        auto callInstanceOfValue = mlirGen(callLogic, genContext);
        return callInstanceOfValue;
    }

    mlir::Value mlirGenInstanceOfLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto result = mlirGen(binaryExpressionAST->left, genContext);
        auto resultType = result.getType();
        auto type = getTypeByTypeName(binaryExpressionAST->right, genContext);

#ifdef ENABLE_RTTI
        if (auto classType = type.dyn_cast_or_null<mlir_ts::ClassType>())
        {
            auto classInfo = getClassInfoByFullName(classType.getName().getValue());
            auto fullNameClassRtti = concat(classInfo->fullName, RTTI_NAME);

            if (resultType.isa<mlir_ts::ClassType>())
            {
                NodeFactory nf(NodeFactoryFlags::None);
                NodeArray<Expression> argumentsArray;
                argumentsArray.push_back(nf.createIdentifier(stows(fullNameClassRtti.str())));
                return mlirGenCallThisMethod(location, result, INSTANCEOF_NAME, undefined, argumentsArray, genContext);
            }

            if (resultType.isa<mlir_ts::AnyType>())
            {
                auto typeOfAnyValue = builder.create<mlir_ts::TypeOfOp>(location, getStringType(), result);
                auto classStrConst = builder.create<mlir_ts::ConstantOp>(location, getStringType(), builder.getStringAttr("class"));
                auto cmpResult = builder.create<mlir_ts::StringCompareOp>(location, getBooleanType(), typeOfAnyValue, classStrConst,
                                                                          builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsToken));

                MLIRCodeLogicHelper mclh(builder, location);
                auto returnValue = mclh.conditionalExpression(
                    getBooleanType(), cmpResult,
                    [&](mlir::OpBuilder &builder, mlir::Location location) {
                        auto thisPtrValue = cast(location, getOpaqueType(), result, genContext);

                        // get VTable we can use VTableOffset
                        auto vtablePtr =
                            builder.create<mlir_ts::VTableOffsetRefOp>(location, getOpaqueType(), thisPtrValue, 0 /*VTABLE index*/);

                        // get InstanceOf method, this is 0 index in vtable
                        auto instanceOfPtr =
                            builder.create<mlir_ts::VTableOffsetRefOp>(location, getOpaqueType(), vtablePtr, 0 /*InstanceOf index*/);

                        auto rttiOfClassValue = resolveFullNameIdentifier(location, fullNameClassRtti, false, genContext);

                        assert(rttiOfClassValue);

                        auto instanceOfFuncType =
                            mlir_ts::FunctionType::get(builder.getContext(), SmallVector<mlir::Type>{getOpaqueType(), getStringType()},
                                                       SmallVector<mlir::Type>{getBooleanType()});

                        auto funcPtr = cast(location, instanceOfFuncType, instanceOfPtr, genContext);

                        // call methos, we need to send, this, and rtti info
                        auto callResult =
                            builder.create<mlir_ts::CallIndirectOp>(location, funcPtr, mlir::ValueRange{thisPtrValue, rttiOfClassValue});

                        return callResult.getResult(0);
                    },
                    [&](mlir::OpBuilder &builder, mlir::Location location) { // default false value
                                                                             // compare typeOfValue
                        return builder.create<mlir_ts::ConstantOp>(location, getBooleanType(), builder.getBoolAttr(false));
                    });

                return returnValue;
            }
        }
#endif

        // default logic
        return builder.create<mlir_ts::ConstantOp>(location, getBooleanType(), builder.getBoolAttr(resultType == type));
    }

    mlir::Value evaluateBinaryOp(mlir::Location location, SyntaxKind opCode, mlir_ts::ConstantOp leftConstOp,
                                 mlir_ts::ConstantOp rightConstOp, const GenContext &genContext)
    {
        auto leftInt = leftConstOp.valueAttr().dyn_cast<mlir::IntegerAttr>().getInt();
        auto rightInt = rightConstOp.valueAttr().dyn_cast<mlir::IntegerAttr>().getInt();
        auto resultType = leftConstOp.getType();

        int64_t result = 0;
        switch (opCode)
        {
        case SyntaxKind::PlusEqualsToken:
            result = leftInt + rightInt;
            break;
        case SyntaxKind::LessThanLessThanToken:
            result = leftInt << rightInt;
            break;
        case SyntaxKind::GreaterThanGreaterThanToken:
            result = leftInt >> rightInt;
            break;
        case SyntaxKind::AmpersandToken:
            result = leftInt & rightInt;
            break;
        case SyntaxKind::BarToken:
            result = leftInt | rightInt;
            break;
        default:
            llvm_unreachable("not implemented");
            break;
        }

        leftConstOp.erase();
        rightConstOp.erase();

        return builder.create<mlir_ts::ConstantOp>(location, resultType, builder.getI64IntegerAttr(result));
    }

    mlir::Value mlirGenSaveLogicOneItem(mlir::Location location, mlir::Value leftExpressionValue, mlir::Value rightExpressionValue,
                                        const GenContext &genContext)
    {
        auto leftExpressionValueBeforeCast = leftExpressionValue;

        if (leftExpressionValue.getType() != rightExpressionValue.getType())
        {
            if (rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
            {
                rightExpressionValue = cast(location, getStringType(), rightExpressionValue, genContext);
            }
        }

        auto savingValue = rightExpressionValue;

        auto syncSavingValue = [&](mlir::Type destType) {
            if (destType != savingValue.getType())
            {
                savingValue = cast(location, destType, savingValue, genContext);
            }
        };

        // TODO: finish it for field access, review CodeLogicHelper.saveResult
        if (auto loadOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::LoadOp>())
        {
            mlir::Type destType;
            TypeSwitch<mlir::Type>(loadOp.reference().getType())
                .Case<mlir_ts::RefType>([&](auto refType) { destType = refType.getElementType(); })
                .Case<mlir_ts::BoundRefType>([&](auto boundRefType) { destType = boundRefType.getElementType(); });

            assert(destType);

            LLVM_DEBUG(llvm::dbgs() << "\n!! Dest type: " << destType << "\n";);

            syncSavingValue(destType);

            // TODO: when saving const array into variable we need to allocate space and copy array as we need to have writable array
            builder.create<mlir_ts::StoreOp>(location, savingValue, loadOp.reference());
        }
        else if (auto accessorOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::AccessorOp>())
        {
            syncSavingValue(accessorOp.getType());

            auto callRes = builder.create<mlir_ts::CallOp>(location, accessorOp.setAccessor().getValue(), mlir::TypeRange{getVoidType()},
                                                           mlir::ValueRange{savingValue});
            savingValue = callRes.getResult(0);
        }
        else if (auto thisAccessorOp = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::ThisAccessorOp>())
        {
            syncSavingValue(thisAccessorOp.getType());

            auto callRes =
                builder.create<mlir_ts::CallOp>(location, thisAccessorOp.setAccessor().getValue(), mlir::TypeRange{getVoidType()},
                                                mlir::ValueRange{thisAccessorOp.thisVal(), savingValue});
            savingValue = callRes.getResult(0);
        }
        /*
        else if (auto createBoundFunction = leftExpressionValueBeforeCast.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            // TODO: i should not allow to change interface
            return mlirGenSaveLogicOneItem(location, createBoundFunction.func(), rightExpressionValue, genContext);
        }
        */
        else
        {
            LLVM_DEBUG(dbgs() << "\n!! left expr.: " << leftExpressionValueBeforeCast << " ...\n";);
            emitError(location, "saving to constant object");
            return mlir::Value();
        }

        return savingValue;
    }

    mlir::Value mlirGenSaveLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        if (leftExpression == SyntaxKind::ArrayLiteralExpression)
        {
            return mlirGenSaveLogicArray(location, leftExpression.as<ArrayLiteralExpression>(), rightExpression, genContext);
        }

        auto leftExpressionValue = mlirGen(leftExpression, genContext);

        VALIDATE(leftExpressionValue, location)

        auto rightExprGenContext = GenContext(genContext);
        if (auto hybridFuncType = leftExpressionValue.getType().dyn_cast<mlir_ts::HybridFunctionType>())
        {
            rightExprGenContext.argTypeDestFuncType = hybridFuncType;
        }
        else if (auto funcType = leftExpressionValue.getType().dyn_cast<mlir_ts::FunctionType>())
        {
            rightExprGenContext.argTypeDestFuncType = funcType;
        }

        auto rightExpressionValue = mlirGen(rightExpression, rightExprGenContext);

        VALIDATE(rightExpressionValue, location)

        return mlirGenSaveLogicOneItem(location, leftExpressionValue, rightExpressionValue, genContext);
    }

    mlir::Value mlirGenSaveLogicArray(mlir::Location location, ArrayLiteralExpression arrayLiteralExpression, Expression rightExpression,
                                      const GenContext &genContext)
    {
        auto rightExpressionValue = mlirGen(rightExpression, genContext);

        VALIDATE(rightExpressionValue, location)

        mlir::Type elementType;
        TypeSwitch<mlir::Type>(rightExpressionValue.getType())
            .Case<mlir_ts::ArrayType>([&](auto arrayType) { elementType = arrayType.getElementType(); })
            .Case<mlir_ts::ConstArrayType>([&](auto constArrayType) { elementType = constArrayType.getElementType(); })
            .Default([](auto type) { llvm_unreachable("not implemented"); });

        auto index = 0;
        for (auto leftItem : arrayLiteralExpression->elements)
        {
            auto leftExpressionValue = mlirGen(leftItem, genContext);

            VALIDATE(leftExpressionValue, location)

            // TODO: unify array access like Property access
            auto indexValue = builder.create<mlir_ts::ConstantOp>(location, builder.getI32Type(), builder.getI32IntegerAttr(index++));

            auto elemRef =
                builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(elementType), rightExpressionValue, indexValue);
            auto rightValue = builder.create<mlir_ts::LoadOp>(location, elementType, elemRef);

            mlirGenSaveLogicOneItem(location, leftExpressionValue, rightValue, genContext);
        }

        // no passing value
        return mlir::Value();
    }

    mlir::Value mlirGen(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto opCode = (SyntaxKind)binaryExpressionAST->operatorToken;

        auto saveResult = MLIRLogicHelper::isNeededToSaveData(opCode);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        if (opCode == SyntaxKind::AmpersandAmpersandToken || opCode == SyntaxKind::BarBarToken)
        {
            return mlirGenAndOrLogic(binaryExpressionAST, genContext, opCode == SyntaxKind::AmpersandAmpersandToken);
        }

        if (opCode == SyntaxKind::QuestionQuestionToken)
        {
            return mlirGenQuestionQuestionLogic(binaryExpressionAST, genContext);
        }

        if (opCode == SyntaxKind::InKeyword)
        {
            return mlirGenInLogic(binaryExpressionAST, genContext);
        }

        if (opCode == SyntaxKind::InstanceOfKeyword)
        {
            return mlirGenInstanceOfLogic(binaryExpressionAST, genContext);
        }

        if (opCode == SyntaxKind::EqualsToken)
        {
            return mlirGenSaveLogic(binaryExpressionAST, genContext);
        }

        auto leftExpressionValue = mlirGen(leftExpression, genContext);
        auto rightExpressionValue = mlirGen(rightExpression, genContext);

        VALIDATE(rightExpressionValue, location)
        VALIDATE(leftExpressionValue, location)

        // check if const expr.
        if (genContext.allowConstEval)
        {
            auto leftConstOp = dyn_cast_or_null<mlir_ts::ConstantOp>(leftExpressionValue.getDefiningOp());
            auto rightConstOp = dyn_cast_or_null<mlir_ts::ConstantOp>(rightExpressionValue.getDefiningOp());
            if (leftConstOp && rightConstOp)
            {
                // try to evaluate
                return evaluateBinaryOp(location, opCode, leftConstOp, rightConstOp, genContext);
            }
        }

        auto leftExpressionValueBeforeCast = leftExpressionValue;
        auto rightExpressionValueBeforeCast = rightExpressionValue;

        // type preprocess
        // TODO: temporary hack
        if (auto leftType = leftExpressionValue.getType().dyn_cast<mlir_ts::LiteralType>())
        {
            leftExpressionValue = cast(loc(leftExpression), leftType.getElementType(), leftExpressionValue, genContext);
        }

        if (auto rightType = rightExpressionValue.getType().dyn_cast<mlir_ts::LiteralType>())
        {
            rightExpressionValue = cast(loc(rightExpression), rightType.getElementType(), rightExpressionValue, genContext);
        }
        // end of hack

        if (leftExpressionValue.getType() != rightExpressionValue.getType())
        {
            // TODO: temporary hack
            if (leftExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
            {
                leftExpressionValue = cast(loc(leftExpression), getStringType(), leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
            {
                rightExpressionValue = cast(loc(rightExpression), getStringType(), rightExpressionValue, genContext);
            }

            // end todo

            if (!MLIRLogicHelper::isLogicOp(opCode))
            {
                // cast from optional<T> type
                if (auto leftOptType = leftExpressionValue.getType().dyn_cast_or_null<mlir_ts::OptionalType>())
                {
                    leftExpressionValue =
                        builder.create<mlir_ts::ValueOp>(loc(leftExpression), leftOptType.getElementType(), leftExpressionValue);
                }

                if (auto rightOptType = rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::OptionalType>())
                {
                    rightExpressionValue =
                        builder.create<mlir_ts::ValueOp>(loc(rightExpression), rightOptType.getElementType(), rightExpressionValue);
                }
            }
        }
        else if (!MLIRLogicHelper::isLogicOp(opCode))
        {
            // special case both are optionals
            if (auto leftOptType = leftExpressionValue.getType().dyn_cast_or_null<mlir_ts::OptionalType>())
            {
                if (auto rightOptType = rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::OptionalType>())
                {
                    leftExpressionValue =
                        builder.create<mlir_ts::ValueOp>(loc(leftExpression), leftOptType.getElementType(), leftExpressionValue);
                    rightExpressionValue =
                        builder.create<mlir_ts::ValueOp>(loc(rightExpression), rightOptType.getElementType(), rightExpressionValue);
                }
            }
        }

        // cast step
        switch (opCode)
        {
        case SyntaxKind::CommaToken:
            // no cast needed
            break;
        case SyntaxKind::LessThanLessThanToken:
        case SyntaxKind::GreaterThanGreaterThanToken:
        case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
            // cast to int
            if (leftExpressionValue.getType() != builder.getI32Type())
            {
                leftExpressionValue = cast(loc(leftExpression), builder.getI32Type(), leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType() != builder.getI32Type())
            {
                rightExpressionValue = cast(loc(rightExpression), builder.getI32Type(), rightExpressionValue, genContext);
            }

            break;
        case SyntaxKind::SlashToken:
        case SyntaxKind::PercentToken:
        case SyntaxKind::AsteriskAsteriskToken:

            if (leftExpressionValue.getType() != getNumberType())
            {
                leftExpressionValue = cast(loc(leftExpression), getNumberType(), leftExpressionValue, genContext);
            }

            if (rightExpressionValue.getType() != getNumberType())
            {
                rightExpressionValue = cast(loc(rightExpression), getNumberType(), rightExpressionValue, genContext);
            }

            break;
        case SyntaxKind::AsteriskToken:
        case SyntaxKind::MinusToken:
        case SyntaxKind::EqualsEqualsToken:
        case SyntaxKind::EqualsEqualsEqualsToken:
        case SyntaxKind::ExclamationEqualsToken:
        case SyntaxKind::ExclamationEqualsEqualsToken:
        case SyntaxKind::GreaterThanToken:
        case SyntaxKind::GreaterThanEqualsToken:
        case SyntaxKind::LessThanToken:
        case SyntaxKind::LessThanEqualsToken:

            if (leftExpressionValue.getType() != rightExpressionValue.getType())
            {
                // cast to base type
                auto hasNumber = leftExpressionValue.getType() == getNumberType() || rightExpressionValue.getType() == getNumberType();
                if (hasNumber)
                {
                    if (leftExpressionValue.getType() != getNumberType())
                    {
                        leftExpressionValue = cast(loc(leftExpression), getNumberType(), leftExpressionValue, genContext);
                    }

                    if (rightExpressionValue.getType() != getNumberType())
                    {
                        rightExpressionValue = cast(loc(rightExpression), getNumberType(), rightExpressionValue, genContext);
                    }
                }
                else
                {
                    auto hasI32 =
                        leftExpressionValue.getType() == builder.getI32Type() || rightExpressionValue.getType() == builder.getI32Type();
                    if (hasI32)
                    {
                        if (leftExpressionValue.getType() != builder.getI32Type())
                        {
                            leftExpressionValue = cast(loc(leftExpression), builder.getI32Type(), leftExpressionValue, genContext);
                        }

                        if (rightExpressionValue.getType() != builder.getI32Type())
                        {
                            rightExpressionValue = cast(loc(rightExpression), builder.getI32Type(), rightExpressionValue, genContext);
                        }
                    }
                }
            }

            break;
        default:
            if (leftExpressionValue.getType() != rightExpressionValue.getType())
            {
                rightExpressionValue = cast(loc(rightExpression), leftExpressionValue.getType(), rightExpressionValue, genContext);
            }

            break;
        }

        auto result = rightExpressionValue;
        switch (opCode)
        {
        case SyntaxKind::EqualsToken:
            // nothing to do;
            assert(false);
            break;
        case SyntaxKind::EqualsEqualsToken:
        case SyntaxKind::EqualsEqualsEqualsToken:
        case SyntaxKind::ExclamationEqualsToken:
        case SyntaxKind::ExclamationEqualsEqualsToken:
        case SyntaxKind::GreaterThanToken:
        case SyntaxKind::GreaterThanEqualsToken:
        case SyntaxKind::LessThanToken:
        case SyntaxKind::LessThanEqualsToken:
            result = builder.create<mlir_ts::LogicalBinaryOp>(location, getBooleanType(), builder.getI32IntegerAttr((int)opCode),
                                                              leftExpressionValue, rightExpressionValue);
            break;
        case SyntaxKind::CommaToken:
            return rightExpressionValue;
        default:
            result = builder.create<mlir_ts::ArithmeticBinaryOp>(
                location, leftExpressionValue.getType(), builder.getI32IntegerAttr((int)opCode), leftExpressionValue, rightExpressionValue);
            break;
        }

        if (saveResult)
        {
            if (leftExpressionValueBeforeCast.getType() != result.getType())
            {
                result = cast(loc(leftExpression), leftExpressionValueBeforeCast.getType(), result, genContext);
            }

            // TODO: finish it for field access, review CodeLogicHelper.saveResult
            if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(leftExpressionValueBeforeCast.getDefiningOp()))
            {
                // TODO: when saving const array into variable we need to allocate space and copy array as we need to have writable
                // array
                builder.create<mlir_ts::StoreOp>(location, result, loadOp.reference());
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        return result;
    }

    mlir::Value mlirGen(SpreadElement spreadElement, const GenContext &genContext)
    {
        return mlirGen(spreadElement->expression, genContext);
    }

    mlir::Value mlirGen(ParenthesizedExpression parenthesizedExpression, const GenContext &genContext)
    {
        return mlirGen(parenthesizedExpression->expression, genContext);
    }

    mlir::Value mlirGen(QualifiedName qualifiedName, const GenContext &genContext)
    {
        auto location = loc(qualifiedName);

        auto expression = qualifiedName->left;
        auto expressionValue = mlirGenModuleReference(expression, genContext);

        VALIDATE(expressionValue, location)

        auto name = MLIRHelper::getName(qualifiedName->right);

        return mlirGenPropertyAccessExpression(location, expressionValue, name, genContext);
    }

    mlir::Value mlirGen(PropertyAccessExpression propertyAccessExpression, const GenContext &genContext)
    {
        auto location = loc(propertyAccessExpression);

        auto expression = propertyAccessExpression->expression.as<Expression>();
        auto expressionValue = mlirGen(expression, genContext);

        VALIDATE(expressionValue, location)

        auto namePtr = MLIRHelper::getName(propertyAccessExpression->name, stringAllocator);

        return mlirGenPropertyAccessExpression(location, expressionValue, namePtr, !!propertyAccessExpression->questionDotToken,
                                               genContext);
    }

    mlir::Value mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue, mlir::StringRef name,
                                                const GenContext &genContext)
    {
        assert(objectValue);
        MLIRPropertyAccessCodeLogic cl(builder, location, objectValue, name);
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, false, cl, genContext);
    }

    mlir::Value mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue, mlir::StringRef name, bool isConditional,
                                                const GenContext &genContext)
    {
        assert(objectValue);
        MLIRPropertyAccessCodeLogic cl(builder, location, objectValue, name);
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, isConditional, cl, genContext);
    }

    mlir::Value mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue, mlir::Attribute id,
                                                const GenContext &genContext)
    {
        MLIRPropertyAccessCodeLogic cl(builder, location, objectValue, id);
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, false, cl, genContext);
    }

    mlir::Value mlirGenPropertyAccessExpression(mlir::Location location, mlir::Value objectValue, mlir::Attribute id, bool isConditional,
                                                const GenContext &genContext)
    {
        MLIRPropertyAccessCodeLogic cl(builder, location, objectValue, id);
        return mlirGenPropertyAccessExpressionLogic(location, objectValue, isConditional, cl, genContext);
    }

    mlir::Value mlirGenPropertyAccessExpressionLogic(mlir::Location location, mlir::Value objectValue, bool isConditional,
                                                     MLIRPropertyAccessCodeLogic &cl, const GenContext &genContext)
    {
        if (isConditional)
        {
            auto condValue = cast(location, getBooleanType(), objectValue, genContext);

            auto propType = evaluateProperty(objectValue, cl.getName().str(), genContext);

            auto ifOp = builder.create<mlir_ts::IfOp>(location, getOptionalType(propType), condValue, true);

            builder.setInsertionPointToStart(&ifOp.thenRegion().front());

            // value if true
            auto value = mlirGenPropertyAccessExpressionBaseLogic(location, objectValue, cl, genContext);
            auto optValue = builder.create<mlir_ts::CreateOptionalOp>(location, getOptionalType(value.getType()), value);
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optValue});

            // else
            builder.setInsertionPointToStart(&ifOp.elseRegion().front());

            auto optUndefValue = builder.create<mlir_ts::UndefOptionalOp>(location, getOptionalType(value.getType()));
            builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optUndefValue});

            builder.setInsertionPointAfter(ifOp);

            return ifOp.results().front();
        }
        else
        {
            return mlirGenPropertyAccessExpressionBaseLogic(location, objectValue, cl, genContext);
        }
    }

    mlir::Value mlirGenPropertyAccessExpressionBaseLogic(mlir::Location location, mlir::Value objectValue, MLIRPropertyAccessCodeLogic &cl,
                                                         const GenContext &genContext)
    {
        mlir::Value value;
        auto name = cl.getName();
        TypeSwitch<mlir::Type>(objectValue.getType())
            .Case<mlir_ts::EnumType>([&](auto enumType) { value = cl.Enum(enumType); })
            .Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { value = cl.Tuple(constTupleType); })
            .Case<mlir_ts::TupleType>([&](auto tupleType) { value = cl.Tuple(tupleType); })
            .Case<mlir_ts::BooleanType>([&](auto intType) { value = cl.Bool(intType); })
            .Case<mlir::IntegerType>([&](auto intType) { value = cl.Int(intType); })
            .Case<mlir::FloatType>([&](auto floatType) { value = cl.Float(floatType); })
            .Case<mlir_ts::NumberType>([&](auto numberType) { value = cl.Number(numberType); })
            .Case<mlir_ts::StringType>([&](auto stringType) { value = cl.String(stringType); })
            .Case<mlir_ts::ConstArrayType>([&](auto arrayType) { value = cl.Array(arrayType); })
            .Case<mlir_ts::ArrayType>([&](auto arrayType) { value = cl.Array(arrayType); })
            .Case<mlir_ts::RefType>([&](auto refType) { value = cl.Ref(refType); })
            .Case<mlir_ts::ObjectType>([&](auto objectType) { value = cl.Object(objectType); })
            .Case<mlir_ts::NamespaceType>([&](auto namespaceType) {
                auto namespaceInfo = getNamespaceByFullName(namespaceType.getName().getValue());
                assert(namespaceInfo);

                auto saveNamespace = currentNamespace;
                currentNamespace = namespaceInfo;

                value = mlirGen(location, name, genContext);

                currentNamespace = saveNamespace;
            })
            .Case<mlir_ts::ClassStorageType>([&](auto classStorageType) {
                value = cl.TupleNoError(classStorageType);
                if (!value)
                {
                    value = ClassMembers(location, objectValue, classStorageType.getName().getValue(), name, true, genContext);
                }
            })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                value = cl.Class(classType);
                if (!value)
                {
                    value = ClassMembers(location, objectValue, classType.getName().getValue(), name, false, genContext);
                }
            })
            .Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                value = InterfaceMembers(location, objectValue, interfaceType.getName().getValue(), cl.getAttribute(), genContext);
            })
            .Case<mlir_ts::OptionalType>([&](auto optionalType) {
                auto elementType = optionalType.getElementType();
                auto loadedValue = builder.create<mlir_ts::ValueOp>(location, elementType, objectValue);
                value = mlirGenPropertyAccessExpression(location, loadedValue, name, false, genContext);
            })
            .Case<mlir_ts::UnionType>([&](auto unionType) {
                // all union types must have the same property
                // 1) cast to first type
                MLIRTypeHelper mth(builder.getContext());
                auto frontType = mth.getFirstNonNullUnionType(unionType);
                auto casted = cast(location, frontType, objectValue, genContext);
                value = mlirGenPropertyAccessExpression(location, casted, name, false, genContext);
            })
            .Default([](auto type) {});

        if (value || genContext.allowPartialResolve)
        {
            return value;
        }

        emitError(location, "Can't resolve property name '") << name << "' of type " << objectValue.getType();

        llvm_unreachable("not implemented");
    }

    mlir::Value ClassMembers(mlir::Location location, mlir::Value thisValue, mlir::StringRef classFullName, mlir::StringRef name,
                             bool baseClass, const GenContext &genContext)
    {
        auto classInfo = getClassInfoByFullName(classFullName);
        assert(classInfo);

        // static field access
        auto value = ClassMembers(location, thisValue, classInfo, name, baseClass, genContext);
        if (!value && !genContext.allowPartialResolve)
        {
            emitError(location, "Class member '") << name << "' can't be found";
        }

        return value;
    }

    mlir::Value ClassMembers(mlir::Location location, mlir::Value thisValue, ClassInfo::TypePtr classInfo, mlir::StringRef name,
                             bool baseClass, const GenContext &genContext)
    {
        assert(classInfo);

        LLVM_DEBUG(llvm::dbgs() << "\n!! looking for member: " << name << " in class '" << classInfo->fullName << "'\n";);

        MLIRCodeLogic mcl(builder);
        auto staticFieldIndex = classInfo->getStaticFieldIndex(mcl.TupleFieldName(name));
        if (staticFieldIndex >= 0)
        {
            auto fieldInfo = classInfo->staticFields[staticFieldIndex];
            auto value = resolveFullNameIdentifier(location, fieldInfo.globalVariableName, false, genContext);
            assert(value);
            return value;
        }

        // check method access
        auto methodIndex = classInfo->getMethodIndex(name);
        if (methodIndex >= 0)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! found method index: " << methodIndex << "\n";);

            auto methodInfo = classInfo->methods[methodIndex];
            auto funcOp = methodInfo.funcOp;
            auto effectiveFuncType = funcOp.getType();

            if (methodInfo.isStatic)
            {
                auto symbOp = builder.create<mlir_ts::SymbolRefOp>(location, effectiveFuncType,
                                                                   mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
                return symbOp;
            }
            else
            {
                auto effectiveThisValue = thisValue;
                if (baseClass)
                {
                    LLVM_DEBUG(dbgs() << "\n!! base call: func '" << funcOp.getName() << "' in context func. '"
                                      << const_cast<GenContext &>(genContext).funcOp.getName() << "', this type: " << thisValue.getType()
                                      << " value:" << thisValue << "\n\n";);

                    // get reference in case of classStorage
                    if (thisValue.getType().isa<mlir_ts::ClassStorageType>())
                    {
                        MLIRCodeLogic mcl(builder);
                        thisValue = mcl.GetReferenceOfLoadOp(thisValue);
                        assert(thisValue);
                    }

                    effectiveThisValue = cast(location, classInfo->classType, thisValue, genContext);
                }

                if (!baseClass && methodInfo.isVirtual)
                {
                    LLVM_DEBUG(dbgs() << "\n!! Virtual call: func '" << funcOp.getName() << "' in context func. '"
                                      << const_cast<GenContext &>(genContext).funcOp.getName() << "'\n";);

                    LLVM_DEBUG(dbgs() << "\n!! Virtual call - this val: [ " << effectiveThisValue << " ] func type: [ " << effectiveFuncType
                                      << " ]\n";);

                    // auto inTheSameFunc = funcOp.getName() == const_cast<GenContext &>(genContext).funcOp.getName();

                    auto vtableAccess = mlirGenPropertyAccessExpression(location, effectiveThisValue, VTABLE_NAME, genContext);

                    assert(genContext.allowPartialResolve || methodInfo.virtualIndex >= 0);

                    auto thisVirtualSymbOp = builder.create<mlir_ts::ThisVirtualSymbolRefOp>(
                        location, getBoundFunctionType(effectiveFuncType), effectiveThisValue, vtableAccess,
                        builder.getI32IntegerAttr(methodInfo.virtualIndex),
                        mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
                    return thisVirtualSymbOp;
                }

                auto thisSymbOp =
                    builder.create<mlir_ts::ThisSymbolRefOp>(location, getBoundFunctionType(effectiveFuncType), effectiveThisValue,
                                                             mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
                return thisSymbOp;
            }
        }

        // check accessor
        auto accessorIndex = classInfo->getAccessorIndex(name);
        if (accessorIndex >= 0)
        {
            auto accessorInfo = classInfo->accessors[accessorIndex];
            auto getFuncOp = accessorInfo.get;
            auto setFuncOp = accessorInfo.set;
            mlir::Type effectiveFuncType;
            if (getFuncOp)
            {
                auto funcType = getFuncOp.getType().dyn_cast<mlir_ts::FunctionType>();
                if (funcType.getNumResults() > 0)
                {
                    effectiveFuncType = funcType.getResult(0);
                }
            }

            if (!effectiveFuncType && setFuncOp)
            {
                effectiveFuncType = setFuncOp.getType().dyn_cast<mlir_ts::FunctionType>().getInput(accessorInfo.isStatic ? 0 : 1);
            }

            if (!effectiveFuncType)
            {
                if (!genContext.allowPartialResolve)
                {
                    emitError(location) << "can't resolve type of property";
                }

                return mlir::Value();
            }

            if (accessorInfo.isStatic)
            {
                auto accessorOp = builder.create<mlir_ts::AccessorOp>(
                    location, effectiveFuncType,
                    getFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), getFuncOp.getName()) : mlir::FlatSymbolRefAttr{},
                    setFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), setFuncOp.getName()) : mlir::FlatSymbolRefAttr{});
                return accessorOp;
            }
            else
            {
                auto thisAccessorOp = builder.create<mlir_ts::ThisAccessorOp>(
                    location, effectiveFuncType, thisValue,
                    getFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), getFuncOp.getName()) : mlir::FlatSymbolRefAttr{},
                    setFuncOp ? mlir::FlatSymbolRefAttr::get(builder.getContext(), setFuncOp.getName()) : mlir::FlatSymbolRefAttr{});
                return thisAccessorOp;
            }
        }

        auto first = true;
        for (auto baseClass : classInfo->baseClasses)
        {
            if (first && name == SUPER_NAME)
            {
                auto value = mlirGenPropertyAccessExpression(location, thisValue, baseClass->fullName, genContext);
                return value;
            }

            auto value = ClassMembers(location, thisValue, baseClass, name, true, genContext);
            if (value)
            {
                return value;
            }

            SmallVector<ClassInfo::TypePtr> fieldPath;
            if (classHasField(baseClass, name, fieldPath))
            {
                // load value from path
                auto currentObject = thisValue;
                for (auto &chain : fieldPath)
                {
                    auto fieldValue = mlirGenPropertyAccessExpression(location, currentObject, chain->fullName, genContext);
                    if (!fieldValue)
                    {
                        if (!genContext.allowPartialResolve)
                        {
                            emitError(location) << "Can't resolve field/property/base '" << chain->fullName << "' of class '"
                                                << classInfo->fullName << "'\n";
                        }

                        return fieldValue;
                    }

                    assert(fieldValue);
                    currentObject = fieldValue;
                }

                // last value
                auto value = mlirGenPropertyAccessExpression(location, currentObject, name, genContext);
                if (value)
                {
                    return value;
                }
            }

            first = false;
        }

        if (baseClass || genContext.allowPartialResolve)
        {
            return mlir::Value();
        }

        emitError(location) << "can't resolve property/field/base '" << name << "' of class '" << classInfo->fullName << "'\n";

        assert(false);
        llvm_unreachable("not implemented");
    }

    bool classHasField(ClassInfo::TypePtr classInfo, mlir::StringRef name, SmallVector<ClassInfo::TypePtr> &fieldPath)
    {
        MLIRCodeLogic mcl(builder);

        auto fieldId = mcl.TupleFieldName(name);
        auto classStorageType = classInfo->classType.getStorageType().cast<mlir_ts::ClassStorageType>();
        auto fieldIndex = classStorageType.getIndex(fieldId);
        auto missingField = fieldIndex < 0 || fieldIndex >= classStorageType.size();
        if (!missingField)
        {
            fieldPath.insert(fieldPath.begin(), classInfo);
            return true;
        }

        for (auto baseClass : classInfo->baseClasses)
        {
            if (classHasField(baseClass, name, fieldPath))
            {
                fieldPath.insert(fieldPath.begin(), classInfo);
                return true;
            }
        }

        return false;
    }

    mlir::Value InterfaceMembers(mlir::Location location, mlir::Value interfaceValue, mlir::StringRef interfaceFullName, mlir::Attribute id,
                                 const GenContext &genContext)
    {
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceFullName);
        assert(interfaceInfo);

        // static field access
        auto value = InterfaceMembers(location, interfaceValue, interfaceInfo, id, genContext);
        if (!value && !genContext.allowPartialResolve)
        {
            emitError(location, "Interface member '") << id << "' can't be found";
        }

        return value;
    }

    mlir::Value InterfaceMembers(mlir::Location location, mlir::Value interfaceValue, InterfaceInfo::TypePtr interfaceInfo,
                                 mlir::Attribute id, const GenContext &genContext)
    {
        assert(interfaceInfo);

        // check field access
        auto totalOffset = 0;
        auto fieldInfo = interfaceInfo->findField(id, totalOffset);
        if (fieldInfo)
        {
            assert(fieldInfo->interfacePosIndex >= 0);
            auto vtableIndex = fieldInfo->interfacePosIndex + totalOffset;

            auto fieldRefType = mlir_ts::RefType::get(fieldInfo->type);

            auto interfaceSymbolRefValue = builder.create<mlir_ts::InterfaceSymbolRefOp>(
                location, fieldRefType, interfaceValue, builder.getI32IntegerAttr(vtableIndex), builder.getStringAttr(""),
                builder.getBoolAttr(fieldInfo->isConditional));

            mlir::Value value;
            if (!fieldInfo->isConditional)
            {
                value = builder.create<mlir_ts::LoadOp>(location, fieldRefType.getElementType(), interfaceSymbolRefValue.getResult());
            }
            else
            {
                auto actualType = fieldRefType.getElementType().isa<mlir_ts::OptionalType>()
                                      ? fieldRefType.getElementType()
                                      : mlir_ts::OptionalType::get(fieldRefType.getElementType());
                value = builder.create<mlir_ts::LoadOp>(location, actualType, interfaceSymbolRefValue.getResult());
            }

            // if it is FuncType, we need to create BoundMethod again
            if (auto funcType = fieldInfo->type.dyn_cast<mlir_ts::FunctionType>())
            {
                auto thisVal = builder.create<mlir_ts::ExtractInterfaceThisOp>(location, getOpaqueType(), interfaceValue);
                value = builder.create<mlir_ts::CreateBoundFunctionOp>(location, getBoundFunctionType(funcType), thisVal, value);
            }

            return value;
        }

        // check method access
        if (auto nameAttr = id.dyn_cast_or_null<mlir::StringAttr>())
        {
            auto name = nameAttr.getValue();
            auto methodInfo = interfaceInfo->findMethod(name, totalOffset);
            if (methodInfo)
            {
                assert(methodInfo->interfacePosIndex >= 0);
                auto vtableIndex = methodInfo->interfacePosIndex + totalOffset;

                auto effectiveFuncType = getBoundFunctionType(methodInfo->funcType);

                auto interfaceSymbolRefValue = builder.create<mlir_ts::InterfaceSymbolRefOp>(
                    location, effectiveFuncType, interfaceValue, builder.getI32IntegerAttr(vtableIndex),
                    builder.getStringAttr(methodInfo->name), builder.getBoolAttr(methodInfo->isConditional));

                return interfaceSymbolRefValue;
            }
        }

        return mlir::Value();
    }

    template <typename T>
    mlir::Value mlirGenElementAccess(mlir::Location location, mlir::Value expression, mlir::Value argumentExpression, T tupleType)
    {
        // get index
        if (auto indexConstOp = dyn_cast_or_null<mlir_ts::ConstantOp>(argumentExpression.getDefiningOp()))
        {
            // this is property access
            MLIRPropertyAccessCodeLogic cl(builder, location, expression, indexConstOp.value());
            return cl.Tuple(tupleType, true);
        }
        else
        {
            llvm_unreachable("not implemented (index)");
        }
    }

    mlir::Value mlirGen(ElementAccessExpression elementAccessExpression, const GenContext &genContext)
    {
        auto location = loc(elementAccessExpression);

        auto expression = mlirGen(elementAccessExpression->expression.as<Expression>(), genContext);
        auto argumentExpression = mlirGen(elementAccessExpression->argumentExpression.as<Expression>(), genContext);

        auto arrayType = expression.getType();

        mlir::Type elementType;
        if (auto arrayTyped = arrayType.dyn_cast_or_null<mlir_ts::ArrayType>())
        {
            elementType = arrayTyped.getElementType();
        }
        else if (auto vectorType = arrayType.dyn_cast_or_null<mlir_ts::ConstArrayType>())
        {
            elementType = vectorType.getElementType();
        }
        else if (arrayType.isa<mlir_ts::StringType>())
        {
            elementType = getCharType();
        }
        else if (auto tupleType = arrayType.dyn_cast_or_null<mlir_ts::TupleType>())
        {
            return mlirGenElementAccess(location, expression, argumentExpression, tupleType);
        }
        else if (auto tupleType = arrayType.dyn_cast_or_null<mlir_ts::ConstTupleType>())
        {
            return mlirGenElementAccess(location, expression, argumentExpression, tupleType);
        }
        else
        {
            emitError(location) << "ElementAccessExpression: " << arrayType;
            llvm_unreachable("not implemented (ElementAccessExpression)");
        }

        auto indexType = argumentExpression.getType();
        auto isAllowableType = indexType.isIntOrIndex() && indexType.getIntOrFloatBitWidth() == 32;
        if (!isAllowableType)
        {
            MLIRTypeHelper mth(builder.getContext());
            argumentExpression = cast(location, mth.getStructIndexType(), argumentExpression, genContext);
        }

        auto elemRef = builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(elementType), expression, argumentExpression);
        return builder.create<mlir_ts::LoadOp>(location, elementType, elemRef);
    }

    mlir::Value mlirGen(CallExpression callExpression, const GenContext &genContext)
    {
        auto location = loc(callExpression);

        auto callExpr = callExpression->expression.as<Expression>();

        auto funcResult = mlirGen(callExpr, genContext);
        if (!funcResult)
        {
            if (genContext.allowPartialResolve)
            {
                return mlir::Value();
            }

            emitError(location, "call expression is empty");

            assert(false);
            return mlir::Value();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! evaluate function: " << funcResult << "\n";);

        SmallVector<mlir::Value, 4> operands;
        if (mlir::failed(mlirGenOperands(callExpression->arguments, operands, funcResult.getType(), genContext)))
        {
            if (!genContext.allowPartialResolve)
            {
                emitError(location) << "Call Method: can't resolve values of all parameters";
            }

            return mlir::Value();
        }

        return mlirGenCallExpression(location, funcResult, callExpression->typeArguments, operands, genContext);
    }

    mlir::Value mlirGenCallExpression(mlir::Location location, mlir::Value funcResult, NodeArray<TypeNode> typeArguments,
                                      SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        GenContext specGenContext(genContext);
        specGenContext.callOperands = operands;

        // get function ref.
        auto specializedFuncRefValue = mlirGenSpecialized(location, funcResult, typeArguments, specGenContext);
        if (!specializedFuncRefValue)
        {
            specializedFuncRefValue = funcResult;
        }

        auto attrName = StringRef(IDENTIFIER_ATTR_NAME);
        auto definingOp = specializedFuncRefValue.getDefiningOp();
        if (isNoneType(specializedFuncRefValue.getType()) && definingOp->hasAttrOfType<mlir::FlatSymbolRefAttr>(attrName))
        {
            // TODO: when you resolve names such as "print", "parseInt" should return names in mlirGen(Identifier)
            auto calleeName = definingOp->getAttrOfType<mlir::FlatSymbolRefAttr>(attrName);
            auto functionName = calleeName.getValue();

            // resolve function
            MLIRCustomMethods cm(builder, location);

            if (auto thisSymbolRefOp = specializedFuncRefValue.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
            {
                // do not remove it, it is needed for custom methods to be called correctly
                operands.insert(operands.begin(), thisSymbolRefOp.thisVal());
            }

            return cm.callMethod(functionName, operands, genContext);
        }

        if (auto optFuncRef = specializedFuncRefValue.getType().dyn_cast<mlir_ts::OptionalType>())
        {
            auto condValue = cast(location, getBooleanType(), specializedFuncRefValue, genContext);

            auto resultType = getReturnTypeFromFuncRef(optFuncRef.getElementType());

            auto ifOp = resultType ? builder.create<mlir_ts::IfOp>(location, getOptionalType(resultType), condValue, true)
                                   : builder.create<mlir_ts::IfOp>(location, condValue, false);

            builder.setInsertionPointToStart(&ifOp.thenRegion().front());

            // value if true

            auto innerFuncRef = builder.create<mlir_ts::ValueOp>(location, optFuncRef.getElementType(), specializedFuncRefValue);

            auto hasReturn = false;
            auto value = mlirGenCall(location, innerFuncRef, operands, hasReturn, genContext);
            if (hasReturn)
            {
                auto optValue = builder.create<mlir_ts::CreateOptionalOp>(location, getOptionalType(value.getType()), value);
                builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optValue});
            }

            if (hasReturn)
            {
                // else
                builder.setInsertionPointToStart(&ifOp.elseRegion().front());

                auto optUndefValue = builder.create<mlir_ts::UndefOptionalOp>(location, getOptionalType(resultType));
                builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{optUndefValue});
            }

            builder.setInsertionPointAfter(ifOp);

            if (resultType)
            {
                return ifOp.results().front();
            }

            return mlir::Value();
        }

        auto hasReturn = false;
        auto value = mlirGenCall(location, specializedFuncRefValue, operands, hasReturn, genContext);
        if (value)
        {
            return value;
        }

        assert(!hasReturn);
        return mlir::Value();
    }

    // TODO: rename and put in helper class
    mlir::Type getReturnTypeFromFuncRef(mlir::Type funcType)
    {
        mlir::Type returnType;

        auto f = [&](auto calledFuncType) {
            auto size = calledFuncType.getResults().size();
            if (size == 1)
            {
                returnType = calledFuncType.getResults().front();
            }
            else if (size > 0)
            {
                llvm_unreachable("not implemented");
            }
        };

        TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { f(calledFuncType); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { f(calledFuncType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { f(calledFuncType); })
            .Case<mlir::NoneType>([&](auto calledFuncType) { returnType = builder.getNoneType(); })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "\n!! getReturnTypeFromFuncRef is not implemented for " << type << "\n";);
                llvm_unreachable("not implemented");
            });

        return returnType;
    }

    // TODO: rename and put in helper class
    mlir::Type getFirstParamFromFuncRef(mlir::Type funcType)
    {
        mlir::Type paramType;

        auto f = [&](auto calledFuncType) { return calledFuncType.getInputs().front(); };

        TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { paramType = f(calledFuncType); })
            .Case<mlir::NoneType>([&](auto calledFuncType) { paramType = builder.getNoneType(); })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "\n!! getFirstParamFromFuncRef is not implemented for " << type << "\n";);
                paramType = builder.getNoneType();
            });

        return paramType;
    }

    // TODO: rename and put in helper class
    mlir::Type getParamsTupleTypeFromFuncRef(mlir::Type funcType)
    {
        mlir::Type paramsType;
        if (!funcType)
        {
            return paramsType;
        }

        auto f = [&](auto calledFuncType) {
            SmallVector<mlir_ts::FieldInfo> fieldInfos;
            for (auto param : calledFuncType.getInputs())
            {
                fieldInfos.push_back({mlir::Attribute(), param});
            }

            return getTupleType(fieldInfos);
        };

        TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir::NoneType>([&](auto calledFuncType) { paramsType = builder.getNoneType(); })
            .Default([&](auto type) {
                LLVM_DEBUG(llvm::dbgs() << "\n!! getParamsTupleTypeFromFuncRef is not implemented for " << type << "\n";);
                paramsType = builder.getNoneType();
            });

        return paramsType;
    }

    bool getVarArgFromFuncRef(mlir::Type funcType)
    {
        bool isVarArg = false;

        LLVM_DEBUG(llvm::dbgs() << "\n!! getVarArgFromFuncRef for " << funcType << "\n";);

        TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { isVarArg = calledFuncType.isVarArg(); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { isVarArg = calledFuncType.isVarArg(); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { isVarArg = calledFuncType.isVarArg(); })
            .Case<mlir::NoneType>([&](auto calledFuncType) {})
            .Default([&](auto type) { LLVM_DEBUG(llvm::dbgs() << "\n!! getVarArgFromFuncRef is not implemented for " << type << "\n";); });

        return isVarArg;
    }

    // TODO: rename and put in helper class
    mlir::Type getOmitThisFunctionTypeFromFuncRef(mlir::Type funcType)
    {
        mlir::Type paramsType;

        auto f = [&](auto calledFuncType) {
            using t = decltype(calledFuncType);
            SmallVector<mlir::Type> newInputTypes;
            if (calledFuncType.getInputs().size() > 0)
            {
                newInputTypes.append(calledFuncType.getInputs().begin() + 1, calledFuncType.getInputs().end());
            }

            auto newType = t::get(builder.getContext(), newInputTypes, calledFuncType.getResults());
            return newType;
        };

        TypeSwitch<mlir::Type>(funcType)
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledFuncType) { paramsType = f(calledFuncType); })
            .Case<mlir::NoneType>([&](auto calledFuncType) { paramsType = builder.getNoneType(); })
            .Default([&](auto type) { llvm_unreachable("not implemented"); });

        return paramsType;
    }

    mlir::Value mlirGenCall(mlir::Location location, mlir::Value funcRefValue, SmallVector<mlir::Value, 4> &operands, bool &hasReturn,
                            const GenContext &genContext)
    {
        mlir::Value value;
        hasReturn = false;
        TypeSwitch<mlir::Type>(funcRefValue.getType())
            .Case<mlir_ts::FunctionType>([&](auto calledFuncType) {
                value = mlirGenCallFunction(location, calledFuncType, funcRefValue, operands, hasReturn, genContext);
            })
            .Case<mlir_ts::HybridFunctionType>([&](auto calledFuncType) {
                value = mlirGenCallFunction(location, calledFuncType, funcRefValue, operands, hasReturn, genContext);
            })
            .Case<mlir_ts::BoundFunctionType>([&](auto calledBoundFuncType) {
                auto calledFuncType = getFunctionType(calledBoundFuncType.getInputs(), calledBoundFuncType.getResults());
                auto thisValue = builder.create<mlir_ts::GetThisOp>(location, calledFuncType.getInput(0), funcRefValue);
                auto unboundFuncRefValue = builder.create<mlir_ts::GetMethodOp>(location, calledFuncType, funcRefValue);
                value = mlirGenCallFunction(location, calledFuncType, unboundFuncRefValue, thisValue, operands, hasReturn, genContext);
            })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                // seems we are calling type constructor
                auto newOp = builder.create<mlir_ts::NewOp>(location, classType, builder.getBoolAttr(true));
                mlirGenCallConstructor(location, classType, newOp, operands, false, true, genContext);
                value = newOp;
            })
            .Case<mlir_ts::ClassStorageType>([&](auto classStorageType) {
                MLIRCodeLogic mcl(builder);
                auto refValue = mcl.GetReferenceOfLoadOp(funcRefValue);
                if (refValue)
                {
                    // seems we are calling type constructor for super()
                    mlirGenCallConstructor(location, classStorageType, refValue, operands, true, false, genContext);
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            })
            .Default([&](auto type) {
                // TODO: this is hack, rewrite it
                // it is not function, so just return value as maybe it has been resolved earlier like in case "<number>.ToString()"
                value = funcRefValue;
            });

        return value;
    }

    template <typename T = mlir_ts::FunctionType>
    mlir::Value mlirGenCallFunction(mlir::Location location, T calledFuncType, mlir::Value funcRefValue,
                                    SmallVector<mlir::Value, 4> &operands, bool &hasReturn, const GenContext &genContext)
    {
        return mlirGenCallFunction(location, calledFuncType, funcRefValue, mlir::Value(), operands, hasReturn, genContext);
    }

    template <typename T = mlir_ts::FunctionType>
    mlir::Value mlirGenCallFunction(mlir::Location location, T calledFuncType, mlir::Value funcRefValue, mlir::Value thisValue,
                                    SmallVector<mlir::Value, 4> &operands, bool &hasReturn, const GenContext &genContext)
    {
        hasReturn = false;
        mlir::Value value;

        if (thisValue)
        {
            operands.insert(operands.begin(), thisValue);
        }

        if (mlir::failed(mlirGenCallOperands(location, operands, calledFuncType.getInputs(), calledFuncType.isVarArg(), genContext)))
        {
            emitError(location) << "Call Method: can't resolve values of all parameters";
        }
        else
        {
            for (auto &oper : operands)
            {
                VALIDATE(oper, location)
            }

            // if last is vararg
            if (calledFuncType.isVarArg())
            {
                SmallVector<mlir::Value, 4> varArgOperands;

                auto fromIndex = calledFuncType.getInputs().size() - 1;
                auto toIndex = operands.size();
                for (auto i = fromIndex; i < toIndex; i++)
                {
                    varArgOperands.push_back(operands[i]);
                }

                operands.pop_back_n(toIndex - fromIndex);

                // create array
                auto array = builder.create<mlir_ts::CreateArrayOp>(location, calledFuncType.getInputs().back(), varArgOperands);
                operands.push_back(array);
            }

            // default call by name
            auto callIndirectOp = builder.create<mlir_ts::CallIndirectOp>(location, funcRefValue, operands);

            if (calledFuncType.getResults().size() > 0)
            {
                value = callIndirectOp.getResult(0);
                hasReturn = true;
            }
        }

        return value;
    }

    mlir::LogicalResult mlirGenCallOperands(mlir::Location location, SmallVector<mlir::Value, 4> &operands,
                                            mlir::ArrayRef<mlir::Type> argFuncTypes, bool isVarArg, const GenContext &genContext)
    {
        int opArgsCount = operands.size();
        int funcArgsCount = argFuncTypes.size();

        if (mlir::failed(mlirGenAdjustOperandTypes(operands, argFuncTypes, isVarArg, genContext)))
        {
            return mlir::failure();
        }

        if (funcArgsCount > opArgsCount)
        {
            // -1 to exclude count params
            for (auto i = (size_t)opArgsCount; i < funcArgsCount; i++)
            {
                if (i == 0)
                {
                    if (auto refType = argFuncTypes[i].dyn_cast<mlir_ts::RefType>())
                    {
                        if (refType.getElementType().isa<mlir_ts::TupleType>())
                        {
                            llvm_unreachable("capture or this ref is not resolved.");
                            return mlir::failure();
                        }
                    }
                }

                operands.push_back(builder.create<mlir_ts::UndefOp>(location, argFuncTypes[i]));
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenOperands(NodeArray<Expression> arguments, SmallVector<mlir::Value, 4> &operands, mlir::Type funcType,
                                        const GenContext &genContext)
    {
        mlir_ts::TupleType tupleTypeWithFuncArgs;
        auto lastArgIndex = operands.size() - 1;
        auto isVarArg = false;
        mlir::Type varArgType;
        auto hasType = false;
        auto tupleParamsType = getParamsTupleTypeFromFuncRef(funcType);
        if (!isNoneType(tupleParamsType))
        {
            hasType = true;
            tupleTypeWithFuncArgs = tupleParamsType.cast<mlir_ts::TupleType>();
            lastArgIndex = tupleTypeWithFuncArgs.getFields().size() - 1;
            isVarArg = getVarArgFromFuncRef(funcType);
            if (isVarArg)
            {
                varArgType = tupleTypeWithFuncArgs.getFields().back().type.cast<mlir_ts::ArrayType>().getElementType();
            }
        }

        auto i = 0;
        for (auto expression : arguments)
        {
            GenContext argGenContext(genContext);
            if (hasType)
            {
                if (isVarArg && i >= lastArgIndex)
                {
                    argGenContext.argTypeDestFuncType = varArgType;
                }
                else
                {
                    argGenContext.argTypeDestFuncType = tupleTypeWithFuncArgs.getFieldInfo(i).type;
                }
            }

            auto value = mlirGen(expression, argGenContext);

            TEST_LOGIC(value)

            operands.push_back(value);

            i++;
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenAdjustOperandTypes(SmallVector<mlir::Value, 4> &operands, mlir::ArrayRef<mlir::Type> argFuncTypes,
                                                  bool isVarArg, const GenContext &genContext)
    {
        auto i = 0; // we need to shift in case of 'this'
        auto lastArgIndex = argFuncTypes.size() - 1;
        mlir::Type varArgType;
        if (isVarArg)
        {
            varArgType = argFuncTypes.back().cast<mlir_ts::ArrayType>().getElementType();
        }

        for (auto value : operands)
        {
            mlir::Type argTypeDestFuncType;
            if (i >= argFuncTypes.size() && !isVarArg)
            {
                emitError(value.getLoc()) << "function does not have enough parameters to accept all arguments, arg #" << i;
                return mlir::failure();
            }

            if (isVarArg && i >= lastArgIndex)
            {
                argTypeDestFuncType = varArgType;
            }
            else
            {
                argTypeDestFuncType = argFuncTypes[i];
            }

            VALIDATE_LOGIC(value, value.getLoc())

            if (value.getType() != argTypeDestFuncType)
            {
                auto castValue = cast(value.getLoc(), argTypeDestFuncType, value, genContext);
                operands[i] = castValue;
            }

            i++;
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenCallConstructor(mlir::Location location, mlir_ts::ClassType classType, mlir::Value thisValue,
                                               SmallVector<mlir::Value, 4> &operands, bool castThisValueToClass, bool setVTable,
                                               const GenContext &genContext)
    {
        if (!classType)
        {
            return mlir::failure();
        }

        // register temp var
        auto classInfo = getClassInfoByFullName(classType.getName().getValue());
        return mlirGenCallConstructor(location, classInfo, thisValue, operands, castThisValueToClass, setVTable, genContext);
    }

    mlir::LogicalResult mlirGenCallConstructor(mlir::Location location, mlir_ts::ClassStorageType classStorageType, mlir::Value thisValue,
                                               SmallVector<mlir::Value, 4> &operands, bool castThisValueToClass, bool setVTable,
                                               const GenContext &genContext)
    {
        if (!classStorageType)
        {
            return mlir::failure();
        }

        // register temp var
        auto classInfo = getClassInfoByFullName(classStorageType.getName().getValue());
        return mlirGenCallConstructor(location, classInfo, thisValue, operands, castThisValueToClass, setVTable, genContext);
    }

    mlir::LogicalResult mlirGenCallConstructor(mlir::Location location, ClassInfo::TypePtr classInfo, mlir::Value thisValue,
                                               SmallVector<mlir::Value, 4> &operands, bool castThisValueToClass, bool setVTable,
                                               const GenContext &genContext)
    {
        assert(classInfo);

        auto virtualTable = classInfo->getHasVirtualTable();
        auto hasConstructor = classInfo->getHasConstructor();
        if (!hasConstructor && !virtualTable)
        {
            return mlir::success();
        }

        auto effectiveThisValue = thisValue;
        if (castThisValueToClass)
        {
            effectiveThisValue = cast(location, classInfo->classType, thisValue, genContext);
        }

        // set virtual table
        if (setVTable && classInfo->getHasVirtualTable())
        {
            auto vtableVal = mlirGenPropertyAccessExpression(location, effectiveThisValue, VTABLE_NAME, genContext);
            MLIRCodeLogic mcl(builder);
            auto vtableRefVal = mcl.GetReferenceOfLoadOp(vtableVal);

            // vtable symbol reference
            auto fullClassVTableFieldName = concat(classInfo->fullName, VTABLE_NAME);
            auto vtableAddress = resolveFullNameIdentifier(location, fullClassVTableFieldName, true, genContext);

            mlir::Value vtableValue;
            if (vtableAddress)
            {
                auto castedValue = cast(location, getOpaqueType(), vtableAddress, genContext);
                vtableValue = castedValue;
            }
            else
            {
                // we will resolve type later
                auto classVTableRefOp =
                    builder.create<mlir_ts::AddressOfOp>(location, getOpaqueType(), fullClassVTableFieldName, ::mlir::IntegerAttr());

                vtableValue = classVTableRefOp;
            }

            builder.create<mlir_ts::StoreOp>(location, vtableValue, vtableRefVal);
        }

        if (classInfo->getHasConstructor())
        {
            auto propAccess = mlirGenPropertyAccessExpression(location, effectiveThisValue, CONSTRUCTOR_NAME, false, genContext);
            if (propAccess)
            {
                bool hasReturn;
                mlirGenCall(location, propAccess, operands, hasReturn, genContext);
            }
            else if (!genContext.allowPartialResolve)
            {
                emitError(location) << "Call Constructor: can't find constructor";
            }
        }

        return mlir::success();
    }

    mlir::Value mlirGen(NewExpression newExpression, const GenContext &genContext)
    {
        MLIRTypeHelper mth(builder.getContext());
        auto location = loc(newExpression);

        // 3 cases, name, index access, method call
        mlir::Type type;
        auto typeExpression = newExpression->expression;
        if (typeExpression == SyntaxKind::Identifier || typeExpression == SyntaxKind::QualifiedName ||
            typeExpression == SyntaxKind::PropertyAccessExpression)
        {
            auto value = mlirGen(typeExpression, newExpression->typeArguments, genContext);

            VALIDATE(value, location);

            type = value.getType();
            type = mth.convertConstTupleTypeToTupleType(type);

            assert(type);

            auto resultType = type;
            if (mth.isValueType(type))
            {
                resultType = getValueRefType(type);
            }

            auto newOp = builder.create<mlir_ts::NewOp>(location, resultType, builder.getBoolAttr(false));

            // evaluate constructor
            mlir::Type tupleParamsType;
            auto funcValueRef = evaluateProperty(newOp, CONSTRUCTOR_NAME, genContext);

            SmallVector<mlir::Value, 4> operands;
            if (mlir::failed(mlirGenOperands(newExpression->arguments, operands, funcValueRef, genContext)))
            {
                if (!genContext.allowPartialResolve)
                {
                    emitError(location) << "Call Method: can't resolve values of all parameters";
                }

                return mlir::Value();
            }

            mlirGenCallConstructor(location, resultType.dyn_cast_or_null<mlir_ts::ClassType>(), newOp, operands, false, true, genContext);
            return newOp;
        }
        else if (typeExpression == SyntaxKind::ElementAccessExpression)
        {
            auto elementAccessExpression = typeExpression.as<ElementAccessExpression>();
            typeExpression = elementAccessExpression->expression;
            type = getTypeByTypeName(typeExpression, genContext);
            type = mth.convertConstTupleTypeToTupleType(type);

            assert(type);

            auto count = mlirGen(elementAccessExpression->argumentExpression, genContext);

            if (count.getType() != builder.getI32Type())
            {
                count = cast(location, builder.getI32Type(), count, genContext);
            }

            auto newArrOp = builder.create<mlir_ts::NewArrayOp>(location, getArrayType(type), count);
            return newArrOp;
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::LogicalResult mlirGen(DeleteExpression deleteExpression, const GenContext &genContext)
    {
        MLIRTypeHelper mth(builder.getContext());
        auto location = loc(deleteExpression);

        auto expr = mlirGen(deleteExpression->expression, genContext);

        if (!expr.getType().isa<mlir_ts::RefType>() && !expr.getType().isa<mlir_ts::ValueRefType>() &&
            !expr.getType().isa<mlir_ts::ClassType>())
        {
            if (auto arrayType = expr.getType().dyn_cast_or_null<mlir_ts::ArrayType>())
            {
                expr = cast(location, mlir_ts::RefType::get(arrayType.getElementType()), expr, genContext);
            }
            else
            {
                llvm_unreachable("not implemented");
            }
        }

        builder.create<mlir_ts::DeleteOp>(location, expr);

        return mlir::success();
    }

    mlir::Value mlirGen(VoidExpression voidExpression, const GenContext &genContext)
    {
        MLIRTypeHelper mth(builder.getContext());
        auto location = loc(voidExpression);

        auto expr = mlirGen(voidExpression->expression, genContext);

        auto value = getUndefined(location);

        return value;
    }

    mlir::Value mlirGen(TypeOfExpression typeOfExpression, const GenContext &genContext)
    {
        auto location = loc(typeOfExpression);

        auto result = mlirGen(typeOfExpression->expression, genContext);
        auto typeOfValue = builder.create<mlir_ts::TypeOfOp>(location, getStringType(), result);
        return typeOfValue;
    }

    mlir::Value mlirGen(NonNullExpression nonNullExpression, const GenContext &genContext)
    {
        return mlirGen(nonNullExpression->expression, genContext);
    }

    mlir::Value mlirGen(TemplateLiteralLikeNode templateExpressionAST, const GenContext &genContext)
    {
        auto location = loc(templateExpressionAST);

        auto stringType = getStringType();
        SmallVector<mlir::Value, 4> strs;

        auto text = convertWideToUTF8(templateExpressionAST->head->rawText);
        auto head = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(text));

        // first string
        strs.push_back(head);
        for (auto span : templateExpressionAST->templateSpans)
        {
            auto expression = span->expression;
            auto exprValue = mlirGen(expression, genContext);

            VALIDATE(exprValue, location)

            if (exprValue.getType() != stringType)
            {
                exprValue = cast(location, stringType, exprValue, genContext);
            }

            // expr value
            strs.push_back(exprValue);

            auto spanText = convertWideToUTF8(span->literal->rawText);
            auto spanValue = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(spanText));

            // text
            strs.push_back(spanValue);
        }

        if (strs.size() <= 1)
        {
            return head;
        }

        auto concatValues = builder.create<mlir_ts::StringConcatOp>(location, stringType, mlir::ArrayRef<mlir::Value>{strs});

        return concatValues;
    }

    mlir::Value mlirGen(TaggedTemplateExpression taggedTemplateExpressionAST, const GenContext &genContext)
    {
        auto location = loc(taggedTemplateExpressionAST);

        auto templateExpressionAST = taggedTemplateExpressionAST->_template;

        SmallVector<mlir::Attribute, 4> strs;
        SmallVector<mlir::Value, 4> vals;

        auto text = convertWideToUTF8(templateExpressionAST->head->rawText);

        // first string
        strs.push_back(getStringAttr(text));
        for (auto span : templateExpressionAST->templateSpans)
        {
            // expr value
            auto expression = span->expression;
            auto exprValue = mlirGen(expression, genContext);

            VALIDATE(exprValue, location)

            vals.push_back(exprValue);

            auto spanText = convertWideToUTF8(span->literal->rawText);
            // text
            strs.push_back(getStringAttr(spanText));
        }

        // tag method
        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), strs);
        auto constStringArray = builder.create<mlir_ts::ConstantOp>(location, getConstArrayType(getStringType(), strs.size()), arrayAttr);

        auto strArrayValue = cast(location, getArrayType(getStringType()), constStringArray, genContext);

        vals.insert(vals.begin(), strArrayValue);

        auto callee = mlirGen(taggedTemplateExpressionAST->tag, genContext);

        ArrayRef<mlir::Type> inputs;

        // cast all params if needed
        if (auto hybridFuncType = callee.getType().cast<mlir_ts::HybridFunctionType>())
        {
            inputs = hybridFuncType.getInputs();
        }
        else if (auto funcType = callee.getType().cast<mlir_ts::FunctionType>())
        {
            inputs = funcType.getInputs();
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        SmallVector<mlir::Value, 4> operands;

        auto i = 0;
        for (auto value : vals)
        {
            if (value.getType() != inputs[i])
            {
                auto castValue = cast(value.getLoc(), inputs[i], value, genContext);
                operands.push_back(castValue);
            }
            else
            {
                operands.push_back(value);
            }

            i++;
        }

        // call
        auto callIndirectOp = builder.create<mlir_ts::CallIndirectOp>(location, callee, operands);

        return callIndirectOp.getResult(0);
    }

    mlir::Value mlirGen(NullLiteral nullLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::NullOp>(loc(nullLiteral), getNullType());
    }

    mlir::Value mlirGen(TrueLiteral trueLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::ConstantOp>(loc(trueLiteral), getBooleanType(), mlir::BoolAttr::get(builder.getContext(), true));
    }

    mlir::Value mlirGen(FalseLiteral falseLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::ConstantOp>(loc(falseLiteral), getBooleanType(), mlir::BoolAttr::get(builder.getContext(), false));
    }

    mlir::Value mlirGen(NumericLiteral numericLiteral, const GenContext &genContext)
    {
        if (numericLiteral->text.find(S(".")) == string::npos)
        {
            try
            {
                return builder.create<mlir_ts::ConstantOp>(loc(numericLiteral), builder.getI32Type(),
                                                           builder.getI32IntegerAttr(to_unsigned_integer(numericLiteral->text)));
            }
            catch (const std::out_of_range &)
            {
                return builder.create<mlir_ts::ConstantOp>(loc(numericLiteral), builder.getI64Type(),
                                                           builder.getI64IntegerAttr(to_bignumber(numericLiteral->text)));
            }
        }
#ifdef NUMBER_F64
        return builder.create<mlir_ts::ConstantOp>(loc(numericLiteral), getNumberType(),
                                                   builder.getF64FloatAttr(to_float(numericLiteral->text)));
#else
        return builder.create<mlir_ts::ConstantOp>(loc(numericLiteral), getNumberType(),
                                                   builder.getF32FloatAttr(to_float(numericLiteral->text)));
#endif
    }

    mlir::Value mlirGen(BigIntLiteral bigIntLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::ConstantOp>(loc(bigIntLiteral), builder.getI64Type(),
                                                   builder.getI64IntegerAttr(to_bignumber(bigIntLiteral->text)));
    }

    mlir::Value mlirGen(ts::StringLiteral stringLiteral, const GenContext &genContext)
    {
        auto text = convertWideToUTF8(stringLiteral->text);

        return builder.create<mlir_ts::ConstantOp>(loc(stringLiteral), getStringType(), getStringAttr(text));
    }

    mlir::Value mlirGen(ts::NoSubstitutionTemplateLiteral noSubstitutionTemplateLiteral, const GenContext &genContext)
    {
        auto text = convertWideToUTF8(noSubstitutionTemplateLiteral->text);

        return builder.create<mlir_ts::ConstantOp>(loc(noSubstitutionTemplateLiteral), getStringType(), getStringAttr(text));
    }

    mlir::Value mlirGenAppendArray(mlir::Location location, mlir::Value arrayDest, mlir::Value arraySrc, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        // register vals
        auto srcArrayVarDecl = std::make_shared<VariableDeclarationDOM>("_src_array_", arraySrc.getType(), location);
        declare(srcArrayVarDecl, arraySrc, genContext);

        auto dstArrayVarDecl = std::make_shared<VariableDeclarationDOM>("_dst_array_", arrayDest.getType(), location);
        dstArrayVarDecl->setReadWriteAccess(true);
        declare(dstArrayVarDecl, arrayDest, genContext);

        NodeFactory nf(NodeFactoryFlags::None);

        auto _src_array_ident = nf.createIdentifier(S("_src_array_"));
        auto _dst_array_ident = nf.createIdentifier(S("_dst_array_"));

        auto _push_ident = nf.createIdentifier(S("push"));

        auto _v_ident = nf.createIdentifier(S("_v_"));

        NodeArray<VariableDeclaration> declarations;
        declarations.push_back(nf.createVariableDeclaration(_v_ident));
        auto declList = nf.createVariableDeclarationList(declarations, NodeFlags::Const);

        // access to push
        auto pushExpr = nf.createPropertyAccessExpression(_dst_array_ident, _push_ident);

        NodeArray<Expression> argumentsArray;
        argumentsArray.push_back(_v_ident);

        auto forOfStat =
            nf.createForOfStatement(undefined, declList, _src_array_ident,
                                    nf.createExpressionStatement(nf.createCallExpression(pushExpr, undefined, argumentsArray)));

        mlirGen(forOfStat, genContext);

        return mlir::Value();
    }

    mlir::Value mlirGen(ts::ArrayLiteralExpression arrayLiteral, const GenContext &genContext)
    {
        auto location = loc(arrayLiteral);

        MLIRTypeHelper mth(builder.getContext());

        // first value
        auto isTuple = false;
        mlir::Type elementType;
        SmallVector<std::tuple<mlir::Type, mlir::Value, bool>> values;
        auto nonConst = false;
        auto spreadElements = false;

        for (auto &item : arrayLiteral->elements)
        {
            auto itemValue = mlirGen(item, genContext);
            if (!itemValue)
            {
                // omitted expression
                continue;
            }

            auto type = itemValue.getType();
            if (item == SyntaxKind::SpreadElement)
            {
                if (auto constArray = type.dyn_cast<mlir_ts::ConstArrayType>())
                {
                    auto constantOp = itemValue.getDefiningOp<mlir_ts::ConstantOp>();
                    auto arrayAttr = constantOp.value().cast<mlir::ArrayAttr>();
                    for (auto val : arrayAttr)
                    {
                        auto newConstVal = builder.create<mlir_ts::ConstantOp>(location, val);
                        values.push_back(std::make_tuple(constArray.getElementType(), newConstVal, false));
                    }
                }
                else if (auto array = type.dyn_cast<mlir_ts::ArrayType>())
                {
                    nonConst = true;
                    spreadElements = true;
                    if (!elementType)
                    {
                        elementType = array.getElementType();
                    }

                    values.push_back(std::make_tuple(array, itemValue, true));
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            }
            else
            {
                values.push_back(std::make_tuple(type, itemValue, false));

                if (!elementType)
                {
                    elementType = type;
                }
                else if (elementType != type)
                {
                    // this is tuple.
                    isTuple = true;
                }
            }
        }

        SmallVector<mlir::Attribute> constValues;
        if (!nonConst)
        {
            for (auto &itemValue : values)
            {
                auto constOp = std::get<1>(itemValue).getDefiningOp<mlir_ts::ConstantOp>();
                if (!constOp)
                {
                    nonConst = true;
                    break;
                }

                constValues.push_back(constOp.valueAttr());
            }
        }

        if (nonConst)
        {
            // non const array
            if (isTuple)
            {
                SmallVector<mlir::Value> arrayValues;
                SmallVector<mlir_ts::FieldInfo> fieldInfos;
                for (auto val : values)
                {
                    fieldInfos.push_back({mlir::Attribute(), std::get<0>(val)});
                    arrayValues.push_back(std::get<1>(val));
                }

                return builder.create<mlir_ts::CreateTupleOp>(location, getTupleType(fieldInfos), arrayValues);
            }

            if (!elementType)
            {
                // in case of empty array
                elementType = getAnyType();
            }

            if (!spreadElements)
            {
                SmallVector<mlir::Value> arrayValues;
                for (auto val : values)
                {
                    arrayValues.push_back(std::get<1>(val));
                }

                auto newArrayOp = builder.create<mlir_ts::CreateArrayOp>(location, getArrayType(elementType), arrayValues);
                return newArrayOp;
            }
            else
            {
                MLIRCustomMethods cm(builder, location);
                SmallVector<mlir::Value> emptyArrayValues;
                auto arrType = getArrayType(elementType);
                auto newArrayOp = builder.create<mlir_ts::CreateArrayOp>(location, arrType, emptyArrayValues);
                auto varArray =
                    builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(arrType), newArrayOp, builder.getBoolAttr(false));

                auto loadedVarArray = builder.create<mlir_ts::LoadOp>(location, arrType, varArray);

                // TODO: push every element into array
                for (auto val : values)
                {
                    if (!std::get<2>(val))
                    {
                        SmallVector<mlir::Value> ops;
                        ops.push_back(loadedVarArray);
                        ops.push_back(std::get<1>(val));
                        cm.mlirGenArrayPush(location, ops);
                    }
                    else
                    {
                        mlirGenAppendArray(location, varArray, std::get<1>(val), genContext);
                    }
                }

                auto loadedVarArray2 = builder.create<mlir_ts::LoadOp>(location, arrType, varArray);
                return loadedVarArray2;
            }
        }
        else
        {
            // recheck types, we know all of them are consts
            isTuple = false;
            elementType = mlir::Type();
            SmallVector<mlir::Type> constTypes;
            for (auto &itemValue : values)
            {
                auto type = mth.convertConstArrayTypeToArrayType(std::get<1>(itemValue).getType());
                constTypes.push_back(type);
                if (!elementType)
                {
                    elementType = type;
                }
                else if (elementType != type)
                {
                    // this is tuple.
                    isTuple = true;
                }
            }

            auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), constValues);
            if (isTuple)
            {
                SmallVector<mlir_ts::FieldInfo> fieldInfos;
                for (auto type : constTypes)
                {
                    fieldInfos.push_back({mlir::Attribute(), type});
                }

                return builder.create<mlir_ts::ConstantOp>(loc(arrayLiteral), getConstTupleType(fieldInfos), arrayAttr);
            }

            if (!elementType)
            {
                // in case of empty array
                elementType = getAnyType();
            }

            return builder.create<mlir_ts::ConstantOp>(loc(arrayLiteral), getConstArrayType(elementType, constValues.size()), arrayAttr);
        }
    }

    mlir::Value mlirGen(ts::ObjectLiteralExpression objectLiteral, const GenContext &genContext)
    {
        // TODO: replace all Opaque with ThisType

        MLIRCodeLogic mcl(builder);
        MLIRTypeHelper mth(builder.getContext());

        // first value
        SmallVector<mlir_ts::FieldInfo> fieldInfos;
        SmallVector<mlir::Attribute> values;
        SmallVector<size_t> methodInfos;
        SmallVector<std::pair<std::string, size_t>> methodInfosWithCaptures;
        SmallVector<std::pair<mlir::Attribute, mlir::Value>> fieldsToSet;

        auto location = loc(objectLiteral);

        auto addFuncFieldInfo = [&](mlir::Attribute fieldId, std::string funcName, mlir_ts::FunctionType funcType) {
            auto type = funcType;

            auto captureVars = getCaptureVarsMap().find(funcName);
            auto hasCaptures = captureVars != getCaptureVarsMap().end();
            if (hasCaptures)
            {
#ifdef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
                values.push_back(mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));
#else
                values.push_back(builder.getUnitAttr());
#endif
            }
            else
            {
                values.push_back(mlir::FlatSymbolRefAttr::get(builder.getContext(), funcName));
            }

            fieldInfos.push_back({fieldId, type});
            if (hasCaptures)
            {
                methodInfosWithCaptures.push_back({funcName, fieldInfos.size() - 1});
            }
            else
            {
                methodInfos.push_back(fieldInfos.size() - 1);
            }
        };

        auto addFieldInfoToArrays = [&](mlir::Attribute fieldId, mlir::Type type) {
            values.push_back(builder.getUnitAttr());
            fieldInfos.push_back({fieldId, type});
        };

        auto addFieldInfo = [&](mlir::Attribute fieldId, mlir::Value itemValue) {
            mlir::Type type;
            mlir::Attribute value;
            if (auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(itemValue.getDefiningOp()))
            {
                value = constOp.valueAttr();
                type = mth.convertConstArrayTypeToArrayType(constOp.getType());
            }
            else if (auto symRefOp = dyn_cast_or_null<mlir_ts::SymbolRefOp>(itemValue.getDefiningOp()))
            {
                value = symRefOp.identifierAttr();
                type = symRefOp.getType();
            }
            else if (auto undefOp = dyn_cast_or_null<mlir_ts::UndefOp>(itemValue.getDefiningOp()))
            {
                value = builder.getUnitAttr();
                type = undefOp.getType();
            }
            else
            {
                value = builder.getUnitAttr();
                type = itemValue.getType();
                fieldsToSet.push_back({fieldId, itemValue});
            }

            values.push_back(value);
            fieldInfos.push_back({fieldId, type});
        };

        auto getFieldIdForProperty = [&](PropertyAssignment &propertyAssignment) {
            auto namePtr = MLIRHelper::getName(propertyAssignment->name, stringAllocator);
            if (namePtr.empty())
            {
                auto value = mlirGen(propertyAssignment->name.as<Expression>(), genContext);
                return mcl.ExtractAttr(value);
            }

            return mcl.TupleFieldName(namePtr);
        };

        auto getFieldIdForShorthandProperty = [&](ShorthandPropertyAssignment &shorthandPropertyAssignment) {
            auto namePtr = MLIRHelper::getName(shorthandPropertyAssignment->name, stringAllocator);
            return mcl.TupleFieldName(namePtr);
        };

        auto getFieldIdForFunctionLike = [&](FunctionLikeDeclarationBase &funcLikeDecl) {
            auto namePtr = MLIRHelper::getName(funcLikeDecl->name, stringAllocator);
            return mcl.TupleFieldName(namePtr);
        };

        auto processFunctionLikeProto = [&](mlir::Attribute fieldId, FunctionLikeDeclarationBase &funcLikeDecl) {
            auto funcName = MLIRHelper::getAnonymousName(loc_check(funcLikeDecl));

            auto funcGenContext = GenContext(genContext);
            funcGenContext.thisType = getObjectType(getConstTupleType(fieldInfos));
            funcGenContext.passResult = nullptr;

            auto funcOpWithFuncProto = mlirGenFunctionPrototype(funcLikeDecl, funcGenContext);
            auto &funcOp = std::get<0>(funcOpWithFuncProto);
            auto &funcProto = std::get<1>(funcOpWithFuncProto);
            auto result = std::get<2>(funcOpWithFuncProto);
            if (!result || !funcOp)
            {
                return;
            }

            // fix this parameter type (taking in account that first type can be captured type)
            auto funcType = funcOp.getType();

            LLVM_DEBUG(llvm::dbgs() << "\n!! Object FuncType: " << funcType << "\n";);
            LLVM_DEBUG(llvm::dbgs() << "\n!! Object FuncType - This: " << funcGenContext.thisType << "\n";);

            // process local vars in this context
            if (funcProto->getHasExtraFields())
            {
                // note: this code needed to store local variables for generators
                auto localVars = getLocalVarsInThisContextMap().find(funcName);
                if (localVars != getLocalVarsInThisContextMap().end())
                {
                    for (auto fieldInfo : localVars->getValue())
                    {
                        addFieldInfoToArrays(fieldInfo.id, fieldInfo.type);
                    }
                }
            }

#ifndef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
            auto capturedType = funcType.getInput(0);
            if (funcProto->getHasCapturedVars())
            {
                // save first param as it is captured vars type, and remove it to fix "this" param
                funcType = getFunctionType(funcType.getInputs().slice(1), funcType.getResults());
                LLVM_DEBUG(llvm::dbgs() << "\n!! Object without captured FuncType: " << funcType << "\n";);
            }
#endif

            // recreate type with "this" param as "any"
            auto newFuncType = mth.getFunctionTypeWithOpaqueThis(funcType, true);
            LLVM_DEBUG(llvm::dbgs() << "\n!! Object with this as opaque: " << newFuncType << "\n";);
#ifndef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
            if (funcProto->getHasCapturedVars())
            {
                newFuncType = mth.getFunctionTypeAddingFirstArgType(newFuncType, capturedType);
                LLVM_DEBUG(llvm::dbgs() << "\n!! Object with this as opaque and returned captured as first: " << newFuncType << "\n";);
            }
#endif

            // place holder
            addFuncFieldInfo(fieldId, funcName, newFuncType);
        };

        auto processFunctionLike = [&](mlir_ts::ObjectType objThis, FunctionLikeDeclarationBase &funcLikeDecl) {
            auto funcGenContext = GenContext(genContext);
            funcGenContext.thisType = objThis;
            funcGenContext.passResult = nullptr;
            funcGenContext.rediscover = true;

            mlir::OpBuilder::InsertionGuard guard(builder);
            auto funcOp = mlirGenFunctionLikeDeclaration(funcLikeDecl, funcGenContext);
        };

        // add all fields
        for (auto &item : objectLiteral->properties)
        {
            mlir::Value itemValue;
            mlir::Attribute fieldId;
            if (item == SyntaxKind::PropertyAssignment)
            {
                auto propertyAssignment = item.as<PropertyAssignment>();
                if (propertyAssignment->initializer == SyntaxKind::FunctionExpression ||
                    propertyAssignment->initializer == SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                itemValue = mlirGen(propertyAssignment->initializer, genContext);

                VALIDATE(itemValue, loc(propertyAssignment->initializer))

                fieldId = getFieldIdForProperty(propertyAssignment);
            }
            else if (item == SyntaxKind::ShorthandPropertyAssignment)
            {
                auto shorthandPropertyAssignment = item.as<ShorthandPropertyAssignment>();
                if (shorthandPropertyAssignment->initializer == SyntaxKind::FunctionExpression ||
                    shorthandPropertyAssignment->initializer == SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                itemValue = mlirGen(shorthandPropertyAssignment->name.as<Expression>(), genContext);

                VALIDATE(itemValue, loc(shorthandPropertyAssignment->name))

                fieldId = getFieldIdForShorthandProperty(shorthandPropertyAssignment);
            }
            else if (item == SyntaxKind::MethodDeclaration)
            {
                continue;
            }
            else if (item == SyntaxKind::SpreadAssignment)
            {
                auto spreadAssignment = item.as<SpreadAssignment>();
                auto tupleValue = mlirGen(spreadAssignment->expression, genContext);

                LLVM_DEBUG(llvm::dbgs() << "\n!! SpreadAssignment value: " << tupleValue << "\n";);

                ::llvm::ArrayRef<mlir_ts::FieldInfo> fields;
                TypeSwitch<mlir::Type>(tupleValue.getType())
                    .template Case<mlir_ts::TupleType>([&](auto tupleType) { fields = tupleType.getFields(); })
                    .template Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { fields = constTupleType.getFields(); })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });

                SmallVector<mlir::Type> types;
                for (auto &field : fields)
                {
                    types.push_back(field.type);
                }

                // deconstruct tuple
                auto res = builder.create<mlir_ts::DeconstructTupleOp>(loc(spreadAssignment), types, tupleValue);

                // read all fields
                for (auto pair : llvm::zip(fields, res.results()))
                {
                    addFieldInfo(std::get<0>(pair).id, std::get<1>(pair));
                }

                continue;
            }
            else
            {
                llvm_unreachable("object literal is not implemented(1)");
            }

            assert(genContext.allowPartialResolve || itemValue);

            addFieldInfo(fieldId, itemValue);
        }

        // process all methods
        for (auto &item : objectLiteral->properties)
        {
            mlir::Attribute fieldId;
            if (item == SyntaxKind::PropertyAssignment)
            {
                auto propertyAssignment = item.as<PropertyAssignment>();
                if (propertyAssignment->initializer != SyntaxKind::FunctionExpression &&
                    propertyAssignment->initializer != SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                auto funcLikeDecl = propertyAssignment->initializer.as<FunctionLikeDeclarationBase>();
                fieldId = getFieldIdForProperty(propertyAssignment);
                processFunctionLikeProto(fieldId, funcLikeDecl);
            }
            else if (item == SyntaxKind::ShorthandPropertyAssignment)
            {
                auto shorthandPropertyAssignment = item.as<ShorthandPropertyAssignment>();
                if (shorthandPropertyAssignment->initializer != SyntaxKind::FunctionExpression &&
                    shorthandPropertyAssignment->initializer != SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                auto funcLikeDecl = shorthandPropertyAssignment->initializer.as<FunctionLikeDeclarationBase>();
                fieldId = getFieldIdForShorthandProperty(shorthandPropertyAssignment);
                processFunctionLikeProto(fieldId, funcLikeDecl);
            }
            else if (item == SyntaxKind::MethodDeclaration)
            {
                auto funcLikeDecl = item.as<FunctionLikeDeclarationBase>();
                fieldId = getFieldIdForFunctionLike(funcLikeDecl);
                processFunctionLikeProto(fieldId, funcLikeDecl);
            }
        }

        // create accum. captures
#ifdef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
        llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> accumulatedCaptureVars;

        for (auto &methodRefWithName : methodInfosWithCaptures)
        {
            auto funcName = std::get<0>(methodRefWithName);
            auto methodRef = std::get<1>(methodRefWithName);
            auto &methodInfo = fieldInfos[methodRef];

            if (auto funcType = methodInfo.type.dyn_cast_or_null<mlir_ts::FunctionType>())
            {
                auto captureVars = getCaptureVarsMap().find(funcName);
                if (captureVars != getCaptureVarsMap().end())
                {
                    // mlirGenResolveCapturedVars
                    for (auto &captureVar : captureVars->getValue())
                    {
                        if (accumulatedCaptureVars.count(captureVar.getKey()) > 0)
                        {
                            assert(accumulatedCaptureVars[captureVar.getKey()] == captureVar.getValue());
                        }

                        accumulatedCaptureVars[captureVar.getKey()] = captureVar.getValue();
                    }
                }
                else
                {
                    assert(false);
                }
            }
        }

        if (accumulatedCaptureVars.size() > 0)
        {
            // add all captured
            SmallVector<mlir::Value> accumulatedCapturedValues;
            if (mlir::failed(mlirGenResolveCapturedVars(location, accumulatedCaptureVars, accumulatedCapturedValues, genContext)))
            {
                return mlir::Value();
            }

            auto capturedValue =
                mlirGenCreateCapture(location, mcl.CaptureType(accumulatedCaptureVars), accumulatedCapturedValues, genContext);
            addFieldInfo(mcl.TupleFieldName(CAPTURED_NAME), capturedValue);
        }
#endif

        // final type
        auto constTupleType = getConstTupleType(fieldInfos);
        auto objThis = getObjectType(constTupleType);

        // process all methods
        for (auto &item : objectLiteral->properties)
        {
            if (item == SyntaxKind::PropertyAssignment)
            {
                auto propertyAssignment = item.as<PropertyAssignment>();
                if (propertyAssignment->initializer != SyntaxKind::FunctionExpression &&
                    propertyAssignment->initializer != SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                auto funcLikeDecl = propertyAssignment->initializer.as<FunctionLikeDeclarationBase>();
                processFunctionLike(objThis, funcLikeDecl);
            }
            else if (item == SyntaxKind::ShorthandPropertyAssignment)
            {
                auto shorthandPropertyAssignment = item.as<ShorthandPropertyAssignment>();
                if (shorthandPropertyAssignment->initializer != SyntaxKind::FunctionExpression &&
                    shorthandPropertyAssignment->initializer != SyntaxKind::ArrowFunction)
                {
                    continue;
                }

                auto funcLikeDecl = shorthandPropertyAssignment->initializer.as<FunctionLikeDeclarationBase>();
                processFunctionLike(objThis, funcLikeDecl);
            }
            else if (item == SyntaxKind::MethodDeclaration)
            {
                auto funcLikeDecl = item.as<FunctionLikeDeclarationBase>();
                processFunctionLike(objThis, funcLikeDecl);
            }
        }

        // fix all method types again
        for (auto &methodRef : methodInfos)
        {
            auto &methodInfo = fieldInfos[methodRef];
            if (auto funcType = methodInfo.type.dyn_cast_or_null<mlir_ts::FunctionType>())
            {
                MLIRTypeHelper mth(builder.getContext());

                methodInfo.type = mth.getFunctionTypeReplaceOpaqueWithThisType(funcType, objThis);
            }
        }

        // fix all method types again and load captured functions
        for (auto &methodRefWithName : methodInfosWithCaptures)
        {
            auto funcName = std::get<0>(methodRefWithName);
            auto methodRef = std::get<1>(methodRefWithName);
            auto &methodInfo = fieldInfos[methodRef];

            if (auto funcType = methodInfo.type.dyn_cast_or_null<mlir_ts::FunctionType>())
            {
                MLIRTypeHelper mth(builder.getContext());
                methodInfo.type = mth.getFunctionTypeReplaceOpaqueWithThisType(funcType, objThis);

#ifndef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
                // TODO: investigate if you can allocate trampolines in heap "change false -> true"
                if (auto trampOp = resolveFunctionWithCapture(location, funcName, funcType, false, genContext))
                {
                    fieldsToSet.push_back({methodInfo.id, trampOp});
                }
                else
                {
                    assert(false);
                }
#endif
            }
        }

        auto constTupleTypeWithReplacedThis = getConstTupleType(fieldInfos);

        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), values);
        auto constantVal = builder.create<mlir_ts::ConstantOp>(loc(objectLiteral), constTupleTypeWithReplacedThis, arrayAttr);
        if (fieldsToSet.empty())
        {
            return constantVal;
        }

        auto tupleType = mth.convertConstTupleTypeToTupleType(constantVal.getType());
        return mlirGenCreateTuple(constantVal.getLoc(), tupleType, constantVal, fieldsToSet, genContext);
    }

    mlir::Value mlirGenCreateTuple(mlir::Location location, mlir::Type tupleType, mlir::Value initValue,
                                   SmallVector<std::pair<mlir::Attribute, mlir::Value>> &fieldsToSet, const GenContext &genContext)
    {
        // we need to cast it to tuple and set values
        auto tupleVar =
            builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(tupleType), initValue, builder.getBoolAttr(false));
        for (auto fieldToSet : fieldsToSet)
        {
            auto location = fieldToSet.second.getLoc();
            auto getField = mlirGenPropertyAccessExpression(location, tupleVar, fieldToSet.first, genContext);

            VALIDATE(fieldToSet.second, location)

            auto savedValue = mlirGenSaveLogicOneItem(location, getField, fieldToSet.second, genContext);
        }

        auto loadedValue = builder.create<mlir_ts::LoadOp>(location, tupleType, tupleVar);
        return loadedValue;
    }

    mlir::Value mlirGen(Identifier identifier, const GenContext &genContext)
    {
        auto location = loc(identifier);

        // resolve name
        auto name = MLIRHelper::getName(identifier);

        // info: can't validate it here, in case of "print" etc
        auto value = mlirGen(location, name, genContext);

        // VALIDATE(value)

        return value;
    }

    mlir::Value resolveIdentifierAsVariable(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        if (name.empty())
        {
            return mlir::Value();
        }

        auto value = symbolTable.lookup(name);
        if (value.second && value.first)
        {
            // begin of logic: outer vars
            auto valueRegion = value.first.getParentRegion();
            auto isOuterVar = false;
            // TODO: review code "valueRegion && valueRegion->getParentOp()" is to support async.execute
            if (genContext.funcOp && valueRegion && valueRegion->getParentOp() /* && valueRegion->getParentOp()->getParentOp()*/)
            {
                auto funcRegion = const_cast<GenContext &>(genContext).funcOp.getCallableRegion();
                isOuterVar = !funcRegion->isAncestor(valueRegion);
            }

            // auto isOuterFunctionScope = value.second->getFuncOp() != genContext.funcOp;
            if (isOuterVar && genContext.passResult)
            {
                LLVM_DEBUG(dbgs() << "\n!! capturing var: [" << value.second->getName() << "] value pair: " << value.first << " type: "
                                  << value.second->getType() << " readwrite: " << value.second->getReadWriteAccess() << "\n\n";);

                genContext.passResult->outerVariables.insert({value.second->getName(), value.second});
            }

            // end of logic: outer vars

            if (!value.second->getReadWriteAccess())
            {
                return value.first;
            }

            LLVM_DEBUG(dbgs() << "\n!! variable: " << name << " type: " << value.first.getType() << "\n");

            // load value if memref
            auto valueType = value.first.getType().cast<mlir_ts::RefType>().getElementType();
            return builder.create<mlir_ts::LoadOp>(value.first.getLoc(), valueType, value.first);
        }

        return mlir::Value();
    }

    mlir::LogicalResult mlirGenResolveCapturedVars(mlir::Location location,
                                                   llvm::StringMap<ts::VariableDeclarationDOM::TypePtr> captureVars,
                                                   SmallVector<mlir::Value> &capturedValues, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        for (auto &item : captureVars)
        {
            auto varValue = mlirGen(location, item.first(), genContext);

            // review capturing by ref.  it should match storage type
            auto refValue = mcl.GetReferenceOfLoadOp(varValue);
            if (refValue)
            {
                capturedValues.push_back(refValue);
                // set var as captures
                if (auto varOp = refValue.getDefiningOp<mlir_ts::VariableOp>())
                {
                    varOp.capturedAttr(builder.getBoolAttr(true));
                }
                else if (auto paramOp = refValue.getDefiningOp<mlir_ts::ParamOp>())
                {
                    paramOp.capturedAttr(builder.getBoolAttr(true));
                }
                else if (auto paramOptOp = refValue.getDefiningOp<mlir_ts::ParamOptionalOp>())
                {
                    paramOptOp.capturedAttr(builder.getBoolAttr(true));
                }
                else
                {
                    // TODO: review it.
                    // find out if u need to ensure that data is captured and belong to VariableOp or ParamOp with captured = true
                    LLVM_DEBUG(llvm::dbgs() << "\n!! var must be captured when loaded from other Op: " << refValue << "\n";);
                    // llvm_unreachable("variable must be captured.");
                }
            }
            else
            {
                // this is not ref, this is const value
                capturedValues.push_back(varValue);
            }
        }

        return mlir::success();
    }

    mlir::Value mlirGenCreateCapture(mlir::Location location, mlir::Type capturedType, SmallVector<mlir::Value> capturedValues,
                                     const GenContext &genContext)
    {
        LLVM_DEBUG(for (auto &val : capturedValues) llvm::dbgs() << "\n!! captured val: " << val << "\n";);

        // add attributes to track which one sent by ref.
        auto captured = builder.create<mlir_ts::CaptureOp>(location, capturedType, capturedValues);
        return captured;
    }

    mlir::Value resolveFunctionWithCapture(mlir::Location location, StringRef name, mlir_ts::FunctionType funcType,
                                           bool allocTrampolineInHeap, const GenContext &genContext)
    {
        // check if required capture of vars
        auto captureVars = getCaptureVarsMap().find(name);
        if (captureVars != getCaptureVarsMap().end())
        {
            auto newFuncType = getFunctionType(funcType.getInputs().slice(1), funcType.getResults());

            auto funcSymbolOp =
                builder.create<mlir_ts::SymbolRefOp>(location, funcType, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));

            LLVM_DEBUG(llvm::dbgs() << "\n!! func with capture: first type: [ " << funcType.getInput(0) << " ], func name: " << name
                                    << "\n");

            SmallVector<mlir::Value> capturedValues;
            if (mlir::failed(mlirGenResolveCapturedVars(location, captureVars->getValue(), capturedValues, genContext)))
            {
                return mlir::Value();
            }

            auto captured = mlirGenCreateCapture(location, funcType.getInput(0), capturedValues, genContext);
#ifndef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION
            return builder.create<mlir_ts::TrampolineOp>(location, newFuncType, funcSymbolOp, captured,
                                                         builder.getBoolAttr(allocTrampolineInHeap));
#else
            auto opaqueTypeValue = cast(location, getOpaqueType(), captured, genContext);
            return builder.create<mlir_ts::CreateBoundFunctionOp>(location, getBoundFunctionType(funcType), opaqueTypeValue, funcSymbolOp);
#endif
        }

        return mlir::Value();
    }

#ifdef REPLACE_TRAMPOLINE_WITH_BOUND_FUNCTION

    mlir::Value resolveFunctionWithCapture(mlir::Location location, StringRef name, mlir_ts::BoundFunctionType boundFuncType,
                                           bool allocTrampolineInHeap, const GenContext &genContext)
    {
        // check if required capture of vars
        auto captureVars = getCaptureVarsMap().find(name);
        if (captureVars != getCaptureVarsMap().end())
        {
            auto funcType = getFunctionType(boundFuncType.getInputs(), boundFuncType.getResults());

            auto funcSymbolOp =
                builder.create<mlir_ts::SymbolRefOp>(location, funcType, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));

            MLIRCodeLogic mcl(builder);
            SmallVector<mlir::Value> capturedValues;
            for (auto &item : captureVars->getValue())
            {
                auto varValue = mlirGen(location, item.first(), genContext);

                // review capturing by ref.  it should match storage type
                auto refValue = mcl.GetReferenceOfLoadOp(varValue);
                if (refValue)
                {
                    capturedValues.push_back(refValue);
                    // set var as captures
                    if (auto varOp = refValue.getDefiningOp<mlir_ts::VariableOp>())
                    {
                        varOp.capturedAttr(builder.getBoolAttr(true));
                    }
                }
                else
                {
                    // this is not ref, this is const value
                    capturedValues.push_back(varValue);
                }
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! func with capture: first type: [ " << boundFuncType.getInput(0) << " ], func name: " << name
                                    << "\n");

            LLVM_DEBUG(for (auto &val : capturedValues) llvm::dbgs() << "\n!! captured val: " << val << "\n";);

            // add attributes to track which one sent by ref.
            auto captured = builder.create<mlir_ts::CaptureOp>(location, boundFuncType.getInput(0), capturedValues);
            auto opaqueTypeValue = cast(location, getOpaqueType(), captured, genContext);
            return builder.create<mlir_ts::CreateBoundFunctionOp>(location, boundFuncType, opaqueTypeValue, funcSymbolOp);
        }

        return mlir::Value();
    }

#endif

    mlir::Value resolveFunctionNameInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // resolving function
        auto fn = getFunctionMap().find(name);
        if (fn != getFunctionMap().end())
        {
            auto funcOp = fn->getValue();
            auto funcType = funcOp.getType();

            if (auto trampOp = resolveFunctionWithCapture(location, funcOp.getName(), funcType, false, genContext))
            {
                return trampOp;
            }

            auto symbOp = builder.create<mlir_ts::SymbolRefOp>(location, funcType,
                                                               mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
            return symbOp;
        }

        return mlir::Value();
    }

    mlir::Value resolveIdentifierInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        auto value = resolveFunctionNameInNamespace(location, name, genContext);
        if (value)
        {
            return value;
        }

        if (getGlobalsMap().count(name))
        {
            auto value = getGlobalsMap().lookup(name);
            return globalVariableAccess(location, value, false, genContext);
        }

        // check if we have enum
        if (getEnumsMap().count(name))
        {
            auto enumTypeInfo = getEnumsMap().lookup(name);
            return builder.create<mlir_ts::ConstantOp>(location, getEnumType(enumTypeInfo.first), enumTypeInfo.second);
        }

        if (getGenericFunctionMap().count(name))
        {
            auto genericFunctionInfo = getGenericFunctionMap().lookup(name);

            auto funcSymbolRef = builder.create<mlir_ts::SymbolRefOp>(
                location, getGenericType(), mlir::FlatSymbolRefAttr::get(builder.getContext(), genericFunctionInfo->name));
            return funcSymbolRef;
        }

        if (getClassesMap().count(name))
        {
            auto classInfo = getClassesMap().lookup(name);
            if (!classInfo->classType)
            {
                if (!genContext.allowPartialResolve)
                {
                    emitError(location) << "can't find class: " << name << "\n";
                }

                return mlir::Value();
            }

            return builder.create<mlir_ts::ClassRefOp>(
                location, classInfo->classType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), classInfo->classType.getName().getValue()));
        }

        if (getGenericClassesMap().count(name))
        {
            auto genericClassInfo = getGenericClassesMap().lookup(name);

            return builder.create<mlir_ts::ClassRefOp>(
                location, genericClassInfo->classType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), genericClassInfo->classType.getName().getValue()));
        }

        if (getInterfacesMap().count(name))
        {
            auto interfaceInfo = getInterfacesMap().lookup(name);
            if (!interfaceInfo->interfaceType)
            {
                if (!genContext.allowPartialResolve)
                {
                    emitError(location) << "can't find interface: " << name << "\n";
                }

                return mlir::Value();
            }

            return builder.create<mlir_ts::InterfaceRefOp>(
                location, interfaceInfo->interfaceType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), interfaceInfo->interfaceType.getName().getValue()));
        }

        if (getGenericInterfacesMap().count(name))
        {
            auto genericInterfaceInfo = getGenericInterfacesMap().lookup(name);

            return builder.create<mlir_ts::InterfaceRefOp>(
                location, genericInterfaceInfo->interfaceType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), genericInterfaceInfo->interfaceType.getName().getValue()));
        }

        if (getTypeAliasMap().count(name))
        {
            auto typeAliasInfo = getTypeAliasMap().lookup(name);
            assert(typeAliasInfo);
            return builder.create<mlir_ts::TypeRefOp>(location, typeAliasInfo);
        }

        if (genContext.typeAliasMap.count(name))
        {
            auto typeAliasInfo = genContext.typeAliasMap.lookup(name);
            assert(typeAliasInfo);
            return builder.create<mlir_ts::TypeRefOp>(location, typeAliasInfo);
        }

        // support generic types
        if (genContext.typeParamsWithArgs.size() > 0)
        {
            auto type = getResolveTypeParameter(name, genContext);
            if (type)
            {
                return builder.create<mlir_ts::TypeRefOp>(location, type);
            }
        }

        if (getNamespaceMap().count(name))
        {
            auto namespaceInfo = getNamespaceMap().lookup(name);
            assert(namespaceInfo);
            auto nsName = mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName);
            return builder.create<mlir_ts::NamespaceRefOp>(location, namespaceInfo->namespaceType, nsName);
        }

        if (getImportEqualsMap().count(name))
        {
            auto fullName = getImportEqualsMap().lookup(name);
            auto namespaceInfo = getNamespaceByFullName(fullName);
            if (namespaceInfo)
            {
                assert(namespaceInfo);
                auto nsName = mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName);
                return builder.create<mlir_ts::NamespaceRefOp>(location, namespaceInfo->namespaceType, nsName);
            }

            auto classInfo = getClassInfoByFullName(fullName);
            if (classInfo)
            {
                return builder.create<mlir_ts::ClassRefOp>(
                    location, classInfo->classType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), classInfo->classType.getName().getValue()));
            }

            auto interfaceInfo = getInterfaceInfoByFullName(fullName);
            if (interfaceInfo)
            {
                return builder.create<mlir_ts::InterfaceRefOp>(
                    location, interfaceInfo->interfaceType,
                    mlir::FlatSymbolRefAttr::get(builder.getContext(), interfaceInfo->interfaceType.getName().getValue()));
            }

            assert(false);
        }

        return mlir::Value();
    }

    mlir::Value resolveFullNameIdentifier(mlir::Location location, StringRef name, bool asAddess, const GenContext &genContext)
    {
        if (fullNameGlobalsMap.count(name))
        {
            auto value = fullNameGlobalsMap.lookup(name);
            return globalVariableAccess(location, value, asAddess, genContext);
        }

        return mlir::Value();
    }

    mlir::Value globalVariableAccess(mlir::Location location, VariableDeclarationDOM::TypePtr value, bool asAddess,
                                     const GenContext &genContext)
    {
        if (!value->getReadWriteAccess() && value->getType().isa<mlir_ts::StringType>())
        {
            // load address of const object in global
            return builder.create<mlir_ts::AddressOfConstStringOp>(location, value->getType(), value->getName());
        }
        else
        {
            auto address = builder.create<mlir_ts::AddressOfOp>(location, mlir_ts::RefType::get(value->getType()), value->getName(),
                                                                ::mlir::IntegerAttr());
            if (asAddess)
            {
                return address;
            }

            return builder.create<mlir_ts::LoadOp>(location, value->getType(), address);
        }
    }

    mlir::Value resolveIdentifier(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // built in types
        if (name == UNDEFINED_NAME)
        {
            return getUndefined(location);
        }

        if (name == INFINITY_NAME)
        {
            return getInfinity(location);
        }

        if (name == NAN_NAME)
        {
            return getNaN(location);
        }

        auto value = resolveIdentifierAsVariable(location, name, genContext);
        if (value)
        {
            return value;
        }

        value = resolveIdentifierInNamespace(location, name, genContext);
        if (value)
        {
            return value;
        }

        auto saveNamespace = currentNamespace;

        // search in outer namespaces
        while (currentNamespace->isFunctionNamespace)
        {
            currentNamespace = currentNamespace->parentNamespace;
            value = resolveIdentifierInNamespace(location, name, genContext);
            if (value)
            {
                currentNamespace = saveNamespace;
                return value;
            }
        }

        // search in root namespace
        currentNamespace = rootNamespace;
        value = resolveIdentifierInNamespace(location, name, genContext);
        currentNamespace = saveNamespace;
        if (value)
        {
            return value;
        }

        // try to resolve 'this' if not resolved yet
        if (genContext.thisType && name == THIS_NAME)
        {
            return builder.create<mlir_ts::ClassRefOp>(
                location, genContext.thisType,
                mlir::FlatSymbolRefAttr::get(builder.getContext(), genContext.thisType.cast<mlir_ts::ClassType>().getName().getValue()));
        }

        if (genContext.thisType && name == SUPER_NAME)
        {
            auto thisValue = mlirGen(location, THIS_NAME, genContext);

            auto classInfo = getClassInfoByFullName(genContext.thisType.cast<mlir_ts::ClassType>().getName().getValue());
            auto baseClassInfo = classInfo->baseClasses.front();

            return mlirGenPropertyAccessExpression(location, thisValue, baseClassInfo->fullName, genContext);
        }

        value = resolveFullNameIdentifier(location, name, false, genContext);
        if (value)
        {
            return value;
        }

        return mlir::Value();
    }

    mlir::Value mlirGen(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        auto value = resolveIdentifier(location, name, genContext);
        if (value)
        {
            return value;
        }

        // unresolved reference (for call for example)
        // TODO: put assert here to see which ref names are not resolved
        auto unresolvedSymbol =
            builder.create<mlir_ts::UnresolvedSymbolRefOp>(location, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
        if (genContext.unresolved)
        {
            genContext.unresolved->push_back(std::make_pair(location, name.str()));
        }

        return unresolvedSymbol;
    }

    TypeParameterDOM::TypePtr processTypeParameter(TypeParameterDeclaration typeParameter, const GenContext &genContext)
    {
        auto namePtr = MLIRHelper::getName(typeParameter->name, stringAllocator);
        if (!namePtr.empty())
        {
            auto typeParameterDOM = std::make_shared<TypeParameterDOM>(namePtr.str());
            if (typeParameter->constraint)
            {
                typeParameterDOM->setConstraint(getType(typeParameter->constraint, genContext));
            }

            return typeParameterDOM;
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    mlir::LogicalResult processTypeParameters(NodeArray<TypeParameterDeclaration> typeParameters,
                                              llvm::SmallVector<TypeParameterDOM::TypePtr> &typeParams, const GenContext &genContext)
    {
        for (auto typeParameter : typeParameters)
        {
            typeParams.push_back(processTypeParameter(typeParameter, genContext));
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(TypeAliasDeclaration typeAliasDeclarationAST, const GenContext &genContext)
    {
        auto namePtr = MLIRHelper::getName(typeAliasDeclarationAST->name, stringAllocator);
        if (!namePtr.empty())
        {
            if (typeAliasDeclarationAST->typeParameters.size() > 0)
            {
                llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
                if (mlir::failed(processTypeParameters(typeAliasDeclarationAST->typeParameters, typeParameters, genContext)))
                {
                    return mlir::failure();
                }

                getGenericTypeAliasMap().insert({namePtr, {typeParameters, typeAliasDeclarationAST->type}});
            }
            else
            {
                auto type = getType(typeAliasDeclarationAST->type, genContext);
                getTypeAliasMap().insert({namePtr, type});
            }

            return mlir::success();
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        return mlir::failure();
    }

    mlir::Value mlirGenModuleReference(Node moduleReference, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)moduleReference;
        if (kind == SyntaxKind::QualifiedName)
        {
            return mlirGen(moduleReference.as<QualifiedName>(), genContext);
        }
        else if (kind == SyntaxKind::Identifier)
        {
            return mlirGen(moduleReference.as<Identifier>(), genContext);
        }

        llvm_unreachable("not implemented");
    }

    mlir::LogicalResult mlirGen(ImportEqualsDeclaration importEqualsDeclarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(importEqualsDeclarationAST->name);
        if (!name.empty())
        {
            auto value = mlirGenModuleReference(importEqualsDeclarationAST->moduleReference, genContext);
            if (auto namespaceOp = value.getDefiningOp<mlir_ts::NamespaceRefOp>())
            {
                getImportEqualsMap().insert({name, namespaceOp.identifier()});
                return mlir::success();
            }
            else if (auto classRefOp = value.getDefiningOp<mlir_ts::ClassRefOp>())
            {
                getImportEqualsMap().insert({name, classRefOp.identifier()});
                return mlir::success();
            }
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGen(EnumDeclaration enumDeclarationAST, const GenContext &genContext)
    {
        auto namePtr = MLIRHelper::getName(enumDeclarationAST->name, stringAllocator);
        if (namePtr.empty())
        {
            llvm_unreachable("not implemented");
            return mlir::failure();
        }

        SmallVector<mlir::NamedAttribute> enumValues;
        int64_t index = 0;
        auto activeBits = 0;
        for (auto enumMember : enumDeclarationAST->members)
        {
            auto memberNamePtr = MLIRHelper::getName(enumMember->name, stringAllocator);
            if (memberNamePtr.empty())
            {
                llvm_unreachable("not implemented");
                return mlir::failure();
            }

            mlir::Attribute enumValueAttr;
            if (enumMember->initializer)
            {
                GenContext enumValueGenContext(genContext);
                enumValueGenContext.allowConstEval = true;
                auto enumValue = mlirGen(enumMember->initializer, enumValueGenContext);
                if (auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(enumValue.getDefiningOp()))
                {
                    enumValueAttr = constOp.valueAttr();
                    if (auto intAttr = enumValueAttr.dyn_cast_or_null<mlir::IntegerAttr>())
                    {
                        index = intAttr.getInt();
                        auto currentActiveBits = (int)intAttr.getValue().getActiveBits();
                        if (currentActiveBits > activeBits)
                        {
                            activeBits = currentActiveBits;
                        }
                    }
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            }
            else
            {
                enumValueAttr = builder.getI32IntegerAttr(index);
            }

            enumValues.push_back({mlir::Identifier::get(memberNamePtr, builder.getContext()), enumValueAttr});
            index++;
        }

        // count used bits
        auto indexUsingBits = std::floor(std::log2(index)) + 1;
        if (indexUsingBits > activeBits)
        {
            activeBits = indexUsingBits;
        }

        // get type by size
        auto bits = 32;
        if (bits < activeBits)
        {
            bits = 64;
            if (bits < activeBits)
            {
                bits = 128;
            }
        }

        auto enumIntType = builder.getIntegerType(bits);
        SmallVector<mlir::NamedAttribute> adjustedEnumValues;
        for (auto enumItem : enumValues)
        {
            if (auto intAttr = enumItem.second.dyn_cast_or_null<mlir::IntegerAttr>())
            {
                adjustedEnumValues.push_back({enumItem.first, mlir::IntegerAttr::get(enumIntType, intAttr.getInt())});
            }
            else
            {
                adjustedEnumValues.push_back(enumItem);
            }
        }

        getEnumsMap().insert({namePtr, std::make_pair(enumIntType, mlir::DictionaryAttr::get(builder.getContext(), adjustedEnumValues))});

        return mlir::success();
    }

    mlir::LogicalResult registerGenericClass(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(classDeclarationAST->name);
        if (!name.empty())
        {
            auto namePtr = StringRef(name).copy(stringAllocator);
            auto fullNamePtr = getFullNamespaceName(namePtr);
            if (fullNameGenericClassesMap.count(fullNamePtr))
            {
                return mlir::success();
            }

            llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
            if (mlir::failed(processTypeParameters(classDeclarationAST->typeParameters, typeParameters, genContext)))
            {
                return mlir::failure();
            }

            // register class
            GenericClassInfo::TypePtr newGenericClassPtr = std::make_shared<GenericClassInfo>();
            newGenericClassPtr->name = namePtr;
            newGenericClassPtr->fullName = fullNamePtr;
            newGenericClassPtr->typeParams = typeParameters;
            newGenericClassPtr->classDeclaration = classDeclarationAST;

            mlirGenClassType(newGenericClassPtr);

            getGenericClassesMap().insert({namePtr, newGenericClassPtr});
            fullNameGenericClassesMap.insert(fullNamePtr, newGenericClassPtr);

            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGen(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        // do not proceed for Generic Interfaces for declaration
        auto isGenericClass = classDeclarationAST->typeParameters.size() > 0;
        if (isGenericClass && genContext.typeParamsWithArgs.size() == 0)
        {
            return registerGenericClass(classDeclarationAST, genContext);
        }

        auto newClassPtr = mlirGenClassInfo(classDeclarationAST, genContext);
        if (!newClassPtr)
        {
            return mlir::failure();
        }

        // do not process specialized class second time;
        if (isGenericClass && genContext.typeParamsWithArgs.size() > 0)
        {
            // TODO: investigate why classType is provided already for class
            if ((genContext.allowPartialResolve && newClassPtr->fullyProcessedAtEvaluation) ||
                (!genContext.allowPartialResolve && newClassPtr->fullyProcessed) || newClassPtr->processingStorageClass)
            {
                return mlir::success();
            }
        }

        auto location = loc(classDeclarationAST);

        newClassPtr->processingStorageClass = true;

        mlirGenClassType(newClassPtr);

        if (mlir::failed(mlirGenClassStorageType(location, classDeclarationAST, newClassPtr, genContext)))
        {
            newClassPtr->processingStorageClass = false;
            return mlir::failure();
        }

        newClassPtr->processingStorageClass = false;

        // go to root
        mlir::OpBuilder::InsertPoint savePoint;
        if (isGenericClass)
        {
            savePoint = builder.saveInsertionPoint();
            builder.setInsertionPointToStart(&theModule.body().front());
        }

        mlirGenClassDefaultConstructor(classDeclarationAST, newClassPtr, genContext);
        mlirGenClassDefaultStaticConstructor(classDeclarationAST, newClassPtr, genContext);

#ifdef ENABLE_RTTI
        mlirGenClassInstanceOfMethod(classDeclarationAST, newClassPtr, genContext);
#endif

        if (mlir::failed(mlirGenClassMembers(location, classDeclarationAST, newClassPtr, genContext)))
        {
            return mlir::failure();
        }

        // generate vtable for interfaces in base class
        if (mlir::failed(mlirGenClassBaseInterfaces(location, newClassPtr, genContext)))
        {
            return mlir::failure();
        }

        // generate vtable for interfaces
        for (auto &heritageClause : classDeclarationAST->heritageClauses)
        {
            if (mlir::failed(mlirGenClassHeritageClauseImplements(classDeclarationAST, newClassPtr, heritageClause, genContext)))
            {
                return mlir::failure();
            }
        }

        mlirGenClassVirtualTableDefinition(location, newClassPtr, genContext);

        /*
        // static fields. must be generated after all non-static methods
        if (mlir::failed(mlirGenClassStaticFields(location, classDeclarationAST, newClassPtr, declareClass, genContext)))
        {
            return mlir::failure();
        }

        // static classes needed to be defined after all methods/fields/vtables generated
        if (mlir::failed(mlirGenClassMembers(location, classDeclarationAST, newClassPtr, declareClass, true, genContext)))
        {
            return mlir::failure();
        }
        */

        if (isGenericClass)
        {
            builder.restoreInsertionPoint(savePoint);
        }

        // if we allow multiple class nodes, do we need to store that ClassLikeDecl. has been processed fully
        if (genContext.allowPartialResolve)
        {
            newClassPtr->fullyProcessedAtEvaluation = true;
        }
        else
        {
            newClassPtr->fullyProcessed = true;
        }

        return mlir::success();
    }

    std::string getSpecializedClassName(GenericClassInfo::TypePtr geneticClassPtr, const GenContext &genContext)
    {
        auto name = geneticClassPtr->fullName.str();
        if (genContext.typeParamsWithArgs.size())
        {
            name.append("<");
            auto next = false;
            for (auto typeParam : geneticClassPtr->typeParams)
            {
                if (next)
                {
                    name.append(",");
                }

                auto type = getResolveTypeParameter(typeParam->getName(), genContext);
                llvm::raw_string_ostream s(name);
                s << type;
                next = false;
            }

            name.append(">");
        }

        return name;
    }

    mlir_ts::ClassType getSpecializationClassType(GenericClassInfo::TypePtr genericClassPtr, const GenContext &genContext)
    {
        auto fullSpecializedClassName = getSpecializedClassName(genericClassPtr, genContext);
        auto classInfoType = getClassInfoByFullName(fullSpecializedClassName);
        return classInfoType->classType;
    }

    ClassInfo::TypePtr mlirGenClassInfo(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        auto name = getNameWithArguments(classDeclarationAST, genContext);
        return mlirGenClassInfo(name, classDeclarationAST, genContext);
    }

    ClassInfo::TypePtr mlirGenClassInfo(std::string name, ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        auto namePtr = StringRef(name).copy(stringAllocator);
        auto fullNamePtr = getFullNamespaceName(namePtr);

        ClassInfo::TypePtr newClassPtr;
        if (fullNameClassesMap.count(fullNamePtr))
        {
            newClassPtr = fullNameClassesMap.lookup(fullNamePtr);
            getClassesMap().insert({namePtr, newClassPtr});
        }
        else
        {
            // register class
            newClassPtr = std::make_shared<ClassInfo>();
            newClassPtr->name = namePtr;
            newClassPtr->fullName = fullNamePtr;
            newClassPtr->isAbstract = hasModifier(classDeclarationAST, SyntaxKind::AbstractKeyword);
            newClassPtr->hasVirtualTable = newClassPtr->isAbstract;

            getClassesMap().insert({namePtr, newClassPtr});
            fullNameClassesMap.insert(fullNamePtr, newClassPtr);
        }

        return newClassPtr;
    }

    template <typename T> mlir::LogicalResult mlirGenClassType(T newClassPtr)
    {
        if (newClassPtr)
        {
            auto classFullNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), newClassPtr->fullName);
            newClassPtr->classType = getClassType(classFullNameSymbol, getClassStorageType(classFullNameSymbol));
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGenClassTypeSetFields(ClassInfo::TypePtr newClassPtr, SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        if (newClassPtr)
        {
            newClassPtr->classType.getStorageType().cast<mlir_ts::ClassStorageType>().setFields(fieldInfos);
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGenClassStorageType(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                                ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        SmallVector<mlir_ts::FieldInfo> fieldInfos;

        // add base classes
        for (auto &heritageClause : classDeclarationAST->heritageClauses)
        {
            if (mlir::failed(mlirGenClassHeritageClause(classDeclarationAST, newClassPtr, heritageClause, fieldInfos, genContext)))
            {
                return mlir::failure();
            }
        }

#if ENABLE_RTTI
        newClassPtr->hasVirtualTable = true;
        mlirGenCustomRTTI(location, classDeclarationAST, newClassPtr, genContext);
#endif

        // non-static first
        for (auto &classMember : classDeclarationAST->members)
        {
            if (mlir::failed(mlirGenClassFieldMember(classDeclarationAST, newClassPtr, classMember, fieldInfos, false, genContext)))
            {
                return mlir::failure();
            }
        }

        if (newClassPtr->getHasVirtualTableVariable())
        {
            MLIRCodeLogic mcl(builder);
            auto fieldId = mcl.TupleFieldName(VTABLE_NAME);
            if (fieldInfos.size() == 0 || fieldInfos.front().id != fieldId)
            {
                fieldInfos.insert(fieldInfos.begin(), {fieldId, getOpaqueType()});
            }
        }

        mlirGenClassTypeSetFields(newClassPtr, fieldInfos);

        if (mlir::failed(mlirGenClassStaticFields(location, classDeclarationAST, newClassPtr, genContext)))
        {
            return mlir::failure();
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassStaticFields(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                                 ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // dummy class, not used, needed to sync code
        // TODO: refactor it
        SmallVector<mlir_ts::FieldInfo> fieldInfos;

        // static second
        // TODO: if I use static method in static field initialization, test if I need process static fields after static methods
        for (auto &classMember : classDeclarationAST->members)
        {
            if (mlir::failed(mlirGenClassFieldMember(classDeclarationAST, newClassPtr, classMember, fieldInfos, true, genContext)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassMembers(mlir::Location location, ClassLikeDeclaration classDeclarationAST,
                                            ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        // clear all flags
        for (auto &classMember : classDeclarationAST->members)
        {
            classMember->processed = false;
        }

        for (auto &classMember : newClassPtr->extraMembers)
        {
            classMember->processed = false;
        }

        // add methods when we have classType
        auto notResolved = 0;
        do
        {
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;

            for (auto &classMember : classDeclarationAST->members)
            {
                if (mlir::failed(mlirGenClassMethodMember(classDeclarationAST, newClassPtr, classMember, genContext)))
                {
                    notResolved++;
                }
            }

            for (auto &classMember : newClassPtr->extraMembers)
            {
                if (mlir::failed(mlirGenClassMethodMember(classDeclarationAST, newClassPtr, classMember, genContext)))
                {
                    notResolved++;
                }
            }

            // repeat if not all resolved
            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                // class can depend on other class declarations
                // theModule.emitError("can't resolve dependencies in class: ") << newClassPtr->name;
                return mlir::failure();
            }

        } while (notResolved > 0);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassHeritageClause(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                   HeritageClause heritageClause, SmallVector<mlir_ts::FieldInfo> &fieldInfos,
                                                   const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);

        if (heritageClause->token == SyntaxKind::ExtendsKeyword)
        {
            auto &baseClassInfos = newClassPtr->baseClasses;

            for (auto &extendingType : heritageClause->types)
            {
                auto baseType = mlirGen(extendingType, genContext);
                TypeSwitch<mlir::Type>(baseType.getType())
                    .template Case<mlir_ts::ClassType>([&](auto baseClassType) {
                        auto baseName = baseClassType.getName().getValue();
                        auto fieldId = mcl.TupleFieldName(baseName);
                        fieldInfos.push_back({fieldId, baseClassType.getStorageType()});

                        auto classInfo = getClassInfoByFullName(baseName);
                        if (std::find(baseClassInfos.begin(), baseClassInfos.end(), classInfo) == baseClassInfos.end())
                        {
                            baseClassInfos.push_back(classInfo);
                        }
                    })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });
            }
            return mlir::success();
        }

        if (heritageClause->token == SyntaxKind::ImplementsKeyword)
        {
            newClassPtr->hasVirtualTable = true;

            auto &interfaceInfos = newClassPtr->implements;

            for (auto &implementingType : heritageClause->types)
            {
                if (implementingType->processed)
                {
                    continue;
                }

                auto ifaceType = mlirGen(implementingType->expression, genContext);
                TypeSwitch<mlir::Type>(ifaceType.getType())
                    .template Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                        interfaceInfos.push_back({interfaceInfo, -1, false});
                        // TODO: it will error
                        // implementingType->processed = true;
                    })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassFieldMember(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                ClassElement classMember, SmallVector<mlir_ts::FieldInfo> &fieldInfos, bool staticOnly,
                                                const GenContext &genContext)
    {
        auto isStatic = hasModifier(classMember, SyntaxKind::StaticKeyword);
        if (staticOnly != isStatic)
        {
            return mlir::success();
        }

        auto location = loc(classMember);

        MLIRCodeLogic mcl(builder);

        auto &staticFieldInfos = newClassPtr->staticFields;

        mlir::Value initValue;
        mlir::Attribute fieldId;
        mlir::Type type;
        StringRef memberNamePtr;

        auto isConstructor = classMember == SyntaxKind::Constructor;
        if (isConstructor)
        {
            if (isStatic)
            {
                newClassPtr->hasStaticConstructor = true;
            }
            else
            {
                newClassPtr->hasConstructor = true;
            }
        }

        auto isAbstract = hasModifier(classMember, SyntaxKind::AbstractKeyword);
        if (isAbstract)
        {
            newClassPtr->hasVirtualTable = true;
        }

        auto isVirtual = (classMember->transformFlags & TransformFlags::ForceVirtual) == TransformFlags::ForceVirtual;
#ifdef ALL_METHODS_VIRTUAL
        isVirtual = !isConstructor;
#endif
        if (isVirtual)
        {
            newClassPtr->hasVirtualTable = true;
        }

        if (classMember == SyntaxKind::PropertyDeclaration)
        {
            // property declaration
            auto propertyDeclaration = classMember.as<PropertyDeclaration>();

            auto memberNamePtr = MLIRHelper::getName(propertyDeclaration->name, stringAllocator);
            if (memberNamePtr.empty())
            {
                llvm_unreachable("not implemented");
                return mlir::failure();
            }

            fieldId = mcl.TupleFieldName(memberNamePtr);

            if (!isStatic)
            {
                auto typeAndInitFlag = evaluateTypeAndInit(propertyDeclaration, genContext);
                type = typeAndInitFlag.first;
                if (typeAndInitFlag.second)
                {
                    newClassPtr->hasInitializers = true;
                }

                LLVM_DEBUG(dbgs() << "\n!! class field: " << fieldId << " type: " << type << "\n\n");

                if (isNoneType(type))
                {
                    return mlir::failure();
                }

                fieldInfos.push_back({fieldId, type});
            }
            else
            {
                // process static field - register global
                auto fullClassStaticFieldName = concat(newClassPtr->fullName, memberNamePtr);
                registerVariable(
                    location, fullClassStaticFieldName, true, VariableClass::Var,
                    [&]() {
                        auto isConst = false;
                        mlir::Type typeInit;
                        evaluate(
                            propertyDeclaration->initializer,
                            [&](mlir::Value val) {
                                typeInit = val.getType();
                                isConst = isConstValue(val);
                            },
                            genContext);
                        if (isConst)
                        {
                            return getTypeAndInit(propertyDeclaration, genContext);
                        }

                        newClassPtr->hasStaticInitializers = true;

                        return getTypeOnly(propertyDeclaration, typeInit, genContext);
                    },
                    genContext);

                if (std::find_if(staticFieldInfos.begin(), staticFieldInfos.end(),
                                 [&](auto staticFld) { return staticFld.id == fieldId; }) == staticFieldInfos.end())
                {
                    staticFieldInfos.push_back({fieldId, fullClassStaticFieldName});
                }
            }
        }

        if (classMember == SyntaxKind::Constructor && !isStatic)
        {
            auto constructorDeclaration = classMember.as<ConstructorDeclaration>();
            for (auto &parameter : constructorDeclaration->parameters)
            {
                auto isPublic = hasModifier(parameter, SyntaxKind::PublicKeyword);
                auto isProtected = hasModifier(parameter, SyntaxKind::ProtectedKeyword);
                auto isPrivate = hasModifier(parameter, SyntaxKind::PrivateKeyword);

                if (!(isPublic || isProtected || isPrivate))
                {
                    continue;
                }

                auto parameterNamePtr = MLIRHelper::getName(parameter->name, stringAllocator);
                if (parameterNamePtr.empty())
                {
                    llvm_unreachable("not implemented");
                    return mlir::failure();
                }

                fieldId = mcl.TupleFieldName(parameterNamePtr);

                auto typeAndInit = getTypeAndInit(parameter, genContext);
                type = typeAndInit.first;

                LLVM_DEBUG(dbgs() << "\n+++ class auto-gen field: " << fieldId << " type: " << type << "\n\n");
                if (isNoneType(type))
                {
                    return mlir::failure();
                }

                fieldInfos.push_back({fieldId, type});
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassDefaultConstructor(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                       const GenContext &genContext)
    {
        // if we do not have constructor but have initializers we need to create empty dummy constructor
        if (newClassPtr->hasInitializers && !newClassPtr->hasConstructor)
        {
            // create constructor
            newClassPtr->hasConstructor = true;

            NodeFactory nf(NodeFactoryFlags::None);

            NodeArray<Statement> statements;

            if (!newClassPtr->baseClasses.empty())
            {
                auto superExpr = nf.createToken(SyntaxKind::SuperKeyword);
                auto callSuper = nf.createCallExpression(superExpr, undefined, undefined);
                statements.push_back(nf.createExpressionStatement(callSuper));
            }

            auto body = nf.createBlock(statements, /*multiLine*/ false);
            auto generatedConstructor = nf.createConstructorDeclaration(undefined, undefined, undefined, body);
            newClassPtr->extraMembers.push_back(generatedConstructor);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassDefaultStaticConstructor(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                             const GenContext &genContext)
    {
        // if we do not have constructor but have initializers we need to create empty dummy constructor
        if (newClassPtr->hasStaticInitializers && !newClassPtr->hasStaticConstructor)
        {
            // create constructor
            newClassPtr->hasStaticConstructor = true;

            NodeFactory nf(NodeFactoryFlags::None);

            NodeArray<Statement> statements;

            auto body = nf.createBlock(statements, /*multiLine*/ false);
            ModifiersArray modifiers;
            modifiers.push_back(nf.createToken(SyntaxKind::StaticKeyword));
            auto generatedConstructor = nf.createConstructorDeclaration(undefined, modifiers, undefined, body);
            newClassPtr->extraMembers.push_back(generatedConstructor);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenCustomRTTI(mlir::Location location, ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                          const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);

        auto fieldId = mcl.TupleFieldName(RTTI_NAME);

        // register global
        auto fullClassStaticFieldName = concat(newClassPtr->fullName, RTTI_NAME);
        registerVariable(
            location, fullClassStaticFieldName, true, VariableClass::Var,
            [&]() {
                auto stringType = getStringType();
                auto init = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(newClassPtr->fullName.str()));
                return std::make_pair(stringType, init);
            },
            genContext);

        auto &staticFieldInfos = newClassPtr->staticFields;
        if (std::find_if(staticFieldInfos.begin(), staticFieldInfos.end(), [&](auto staticFld) { return staticFld.id == fieldId; }) ==
            staticFieldInfos.end())
        {
            staticFieldInfos.push_back({fieldId, fullClassStaticFieldName});
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassInstanceOfMethod(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                     const GenContext &genContext)
    {
        // if we do not have constructor but have initializers we need to create empty dummy constructor
        // if (newClassPtr->getHasVirtualTable())
        {
            if (newClassPtr->hasRTTI)
            {
                return mlir::success();
            }

            NodeFactory nf(NodeFactoryFlags::None);

            NodeArray<Statement> statements;

            /*
            if (!newClassPtr->baseClasses.empty())
            {
                auto superExpr = nf.createToken(SyntaxKind::SuperKeyword);
                auto callSuper = nf.createCallExpression(superExpr, undefined, undefined);
                statements.push_back(nf.createExpressionStatement(callSuper));
            }
            */

            // temp return false;
            auto cmpRttiToParam = nf.createBinaryExpression(
                nf.createIdentifier(LINSTANCEOF_PARAM_NAME), nf.createToken(SyntaxKind::EqualsEqualsToken),
                nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::ThisKeyword), nf.createIdentifier(S(RTTI_NAME))));

            auto cmpLogic = cmpRttiToParam;

            if (!newClassPtr->baseClasses.empty())
            {
                NodeArray<Expression> argumentsArray;
                argumentsArray.push_back(nf.createIdentifier(LINSTANCEOF_PARAM_NAME));
                cmpLogic = nf.createBinaryExpression(
                    cmpRttiToParam, nf.createToken(SyntaxKind::BarBarEqualsToken),
                    nf.createCallExpression(
                        nf.createPropertyAccessExpression(nf.createToken(SyntaxKind::SuperKeyword), nf.createIdentifier(LINSTANCEOF_NAME)),
                        undefined, argumentsArray));
            }

            auto returnStat = nf.createReturnStatement(cmpLogic);
            statements.push_back(returnStat);

            auto body = nf.createBlock(statements, false);

            NodeArray<ParameterDeclaration> parameters;
            parameters.push_back(nf.createParameterDeclaration(undefined, undefined, undefined, nf.createIdentifier(LINSTANCEOF_PARAM_NAME),
                                                               undefined, nf.createToken(SyntaxKind::StringKeyword), undefined));

            auto instanceOfMethod =
                nf.createMethodDeclaration(undefined, undefined, undefined, nf.createIdentifier(LINSTANCEOF_NAME), undefined, undefined,
                                           parameters, nf.createToken(SyntaxKind::BooleanKeyword), body);
            instanceOfMethod->transformFlags |= TransformFlags::ForceVirtual;
            // TODO: you adding new member to the same DOM(parse) instance but it is used for 2 instances of generic type
            // ERROR: do not change members!!!!
            newClassPtr->extraMembers.push_back(instanceOfMethod);

            newClassPtr->hasRTTI = true;
        }

        return mlir::success();
    }

    mlir::Value mlirGenCreateInterfaceVTableForClass(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                     InterfaceInfo::TypePtr newInterfacePtr, const GenContext &genContext)
    {
        auto fullClassInterfaceVTableFieldName = interfaceVTableNameForClass(newClassPtr, newInterfacePtr);
        auto existValue = resolveFullNameIdentifier(location, fullClassInterfaceVTableFieldName, true, genContext);
        if (existValue)
        {
            return existValue;
        }

        if (mlir::succeeded(mlirGenClassVirtualTableDefinitionForInterface(location, newClassPtr, newInterfacePtr, genContext)))
        {
            return resolveFullNameIdentifier(location, fullClassInterfaceVTableFieldName, true, genContext);
        }

        return mlir::Value();
    }

    mlir::Value mlirGenCreateInterfaceVTableForObject(mlir::Location location, mlir_ts::ObjectType objectType,
                                                      InterfaceInfo::TypePtr newInterfacePtr, const GenContext &genContext)
    {
        auto fullObjectInterfaceVTableFieldName = interfaceVTableNameForObject(objectType, newInterfacePtr);
        auto existValue = resolveFullNameIdentifier(location, fullObjectInterfaceVTableFieldName, true, genContext);
        if (existValue)
        {
            return existValue;
        }

        if (mlir::succeeded(mlirGenObjectVirtualTableDefinitionForInterface(location, objectType, newInterfacePtr, genContext)))
        {
            return resolveFullNameIdentifier(location, fullObjectInterfaceVTableFieldName, true, genContext);
        }

        return mlir::Value();
    }

    StringRef interfaceVTableNameForClass(ClassInfo::TypePtr newClassPtr, InterfaceInfo::TypePtr newInterfacePtr)
    {
        return concat(newClassPtr->fullName, newInterfacePtr->fullName, VTABLE_NAME);
    }

    StringRef interfaceVTableNameForObject(mlir_ts::ObjectType objectType, InterfaceInfo::TypePtr newInterfacePtr)
    {
        std::stringstream ss;
        ss << hash_value(objectType);

        return concat(newInterfacePtr->fullName, ss.str().c_str(), VTABLE_NAME);
    }

    mlir::LogicalResult canCastTupleToInterface(mlir_ts::TupleType tupleStorageType, InterfaceInfo::TypePtr newInterfacePtr)
    {
        SmallVector<VirtualMethodOrFieldInfo> virtualTable;
        auto location = loc(TextRange());
        return getInterfaceVirtualTableForObject(location, tupleStorageType, newInterfacePtr, virtualTable, true);
    }

    mlir::LogicalResult getInterfaceVirtualTableForObject(mlir::Location location, mlir_ts::TupleType tupleStorageType,
                                                          InterfaceInfo::TypePtr newInterfacePtr,
                                                          SmallVector<VirtualMethodOrFieldInfo> &virtualTable, bool suppressErrors = false)
    {
        MLIRTypeHelper mth(builder.getContext());

        MethodInfo emptyMethod;
        mlir_ts::FieldInfo emptyFieldInfo;
        mlir_ts::FieldInfo missingFieldInfo;

        auto result = newInterfacePtr->getVirtualTable(
            virtualTable,
            [&](mlir::Attribute id, mlir::Type fieldType, bool isConditional) -> mlir_ts::FieldInfo {
                auto foundIndex = tupleStorageType.getIndex(id);
                if (foundIndex >= 0)
                {
                    auto foundField = tupleStorageType.getFieldInfo(foundIndex);
                    auto test =
                        foundField.type.isa<mlir_ts::FunctionType>() && fieldType.isa<mlir_ts::FunctionType>()
                            ? mth.TestFunctionTypesMatchWithObjectMethods(foundField.type, fieldType).result == MatchResultType::Match
                            : fieldType == foundField.type;
                    if (!test)
                    {
                        if (!suppressErrors)
                        {
                            emitError(location)
                                << "field " << id << " not matching type: " << fieldType << " and " << foundField.type << " in interface '"
                                << newInterfacePtr->fullName << "' for object '" << tupleStorageType << "'";
                        }

                        return emptyFieldInfo;
                    }

                    return foundField;
                }

                if (!isConditional)
                {
                    emitError(location) << "field can't be found " << id << " for interface '" << newInterfacePtr->fullName
                                        << "' in object '" << tupleStorageType << "'";
                }

                return emptyFieldInfo;
            },
            [&](std::string name, mlir_ts::FunctionType funcType, bool isConditional) -> MethodInfo & {
                llvm_unreachable("not implemented yet");
            });

        return result;
    }

    mlir::LogicalResult mlirGenObjectVirtualTableDefinitionForInterface(mlir::Location location, mlir_ts::ObjectType objectType,
                                                                        InterfaceInfo::TypePtr newInterfacePtr,
                                                                        const GenContext &genContext)
    {
        MLIRTypeHelper mth(builder.getContext());
        MLIRCodeLogic mcl(builder);

        auto storeType = objectType.getStorageType();
        auto tupleStorageType = mth.convertConstTupleTypeToTupleType(storeType).cast<mlir_ts::TupleType>();

        SmallVector<VirtualMethodOrFieldInfo> virtualTable;
        auto result = getInterfaceVirtualTableForObject(location, tupleStorageType, newInterfacePtr, virtualTable);
        if (mlir::failed(result))
        {
            return result;
        }

        // register global
        auto fullClassInterfaceVTableFieldName = interfaceVTableNameForObject(objectType, newInterfacePtr);
        registerVariable(
            location, fullClassInterfaceVTableFieldName, true, VariableClass::Var,
            [&]() {
                // build vtable from names of methods

                auto virtTuple = getVirtualTableType(virtualTable);

                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto methodOrField : virtualTable)
                {
                    if (methodOrField.isField)
                    {
                        auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                        if (!methodOrField.isMissing)
                        {
                            auto objectNull = cast(location, objectType, nullObj, genContext);
                            auto fieldValue = mlirGenPropertyAccessExpression(location, objectNull, methodOrField.fieldInfo.id, genContext);
                            assert(fieldValue);
                            auto fieldRef = mcl.GetReferenceOfLoadOp(fieldValue);

                            LLVM_DEBUG(llvm::dbgs() << "\n!! vtable field: " << methodOrField.fieldInfo.id << " type: "
                                                    << methodOrField.fieldInfo.type << " provided data: " << fieldRef << "\n";);

                            if (fieldRef.getType().isa<mlir_ts::BoundRefType>())
                            {
                                fieldRef = cast(location, mlir_ts::RefType::get(methodOrField.fieldInfo.type), fieldRef, genContext);
                            }
                            else
                            {
                                assert(fieldRef.getType().cast<mlir_ts::RefType>().getElementType() == methodOrField.fieldInfo.type);
                            }

                            // insert &(null)->field
                            vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                                location, virtTuple, fieldRef, vtableValue, builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex)));
                        }
                        else
                        {
                            // null value, as missing field/method
                            // auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                            auto negative1 = builder.create<mlir_ts::ConstantOp>(location, builder.getI64Type(), mth.getI64AttrValue(-1));
                            auto castedNull = cast(location, mlir_ts::RefType::get(methodOrField.fieldInfo.type), negative1, genContext);
                            vtableValue =
                                builder.create<mlir_ts::InsertPropertyOp>(location, virtTuple, castedNull, vtableValue,
                                                                          builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex)));
                        }
                    }
                    else
                    {
                        llvm_unreachable("not implemented yet");
                        /*#
                        ]]]]]]]]]
                        auto methodConstName = builder.create<mlir_ts::SymbolRefOp>(
                            location, methodOrField.methodInfo.funcOp.getType(),
                            mlir::FlatSymbolRefAttr::get(builder.getContext(), methodOrField.methodInfo.funcOp.sym_name()));

                        vtableValue =
                            builder.create<mlir_ts::InsertPropertyOp>(location, virtTuple, methodConstName, vtableValue,
                                                                      builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex)));
                        */
                    }

                    fieldIndex++;
                }

                return std::pair<mlir::Type, mlir::Value>{virtTuple, vtableValue};
            },
            genContext);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassVirtualTableDefinitionForInterface(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                                       InterfaceInfo::TypePtr newInterfacePtr, const GenContext &genContext)
    {
        MLIRTypeHelper mth(builder.getContext());
        MLIRCodeLogic mcl(builder);

        MethodInfo emptyMethod;
        mlir_ts::FieldInfo emptyFieldInfo;
        // TODO: ...
        auto classStorageType = newClassPtr->classType.getStorageType().cast<mlir_ts::ClassStorageType>();

        llvm::SmallVector<VirtualMethodOrFieldInfo> virtualTable;
        auto result = newInterfacePtr->getVirtualTable(
            virtualTable,
            [&](mlir::Attribute id, mlir::Type fieldType, bool isConditional) -> mlir_ts::FieldInfo {
                auto found = false;
                auto foundField = newClassPtr->findField(id, found);
                if (!found || fieldType != foundField.type)
                {
                    if (!found && !isConditional || found)
                    {
                        emitError(location) << "field type not matching for '" << id << "' for interface '" << newInterfacePtr->fullName
                                            << "' in class '" << newClassPtr->fullName << "'";
                    }

                    return emptyFieldInfo;
                }

                return foundField;
            },
            [&](std::string name, mlir_ts::FunctionType funcType, bool isConditional) -> MethodInfo & {
                auto foundMethodPtr = newClassPtr->findMethod(name);
                if (!foundMethodPtr)
                {
                    if (!isConditional)
                    {
                        emitError(location) << "can't find method '" << name << "' for interface '" << newInterfacePtr->fullName
                                            << "' in class '" << newClassPtr->fullName << "'";
                    }

                    return emptyMethod;
                }

                auto foundMethodFunctionType = foundMethodPtr->funcOp.getType().cast<mlir_ts::FunctionType>();

                auto result = mth.TestFunctionTypesMatch(funcType, foundMethodFunctionType, 1);
                if (result.result != MatchResultType::Match)
                {
                    emitError(location) << "method signature not matching for '" << name << "'{" << funcType << "} for interface '"
                                        << newInterfacePtr->fullName << "' in class '" << newClassPtr->fullName << "'"
                                        << " found method: " << foundMethodFunctionType;

                    return emptyMethod;
                }

                return *foundMethodPtr;
            });

        if (mlir::failed(result))
        {
            return result;
        }

        // register global
        auto fullClassInterfaceVTableFieldName = interfaceVTableNameForClass(newClassPtr, newInterfacePtr);
        registerVariable(
            location, fullClassInterfaceVTableFieldName, true, VariableClass::Var,
            [&]() {
                // build vtable from names of methods

                MLIRCodeLogic mcl(builder);

                auto virtTuple = getVirtualTableType(virtualTable);

                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto methodOrField : virtualTable)
                {
                    if (methodOrField.isField)
                    {
                        auto nullObj = builder.create<mlir_ts::NullOp>(location, getNullType());
                        auto classNull = cast(location, newClassPtr->classType, nullObj, genContext);
                        auto fieldValue = mlirGenPropertyAccessExpression(location, classNull, methodOrField.fieldInfo.id, genContext);
                        auto fieldRef = mcl.GetReferenceOfLoadOp(fieldValue);

                        // insert &(null)->field
                        vtableValue = builder.create<mlir_ts::InsertPropertyOp>(
                            location, virtTuple, fieldRef, vtableValue, builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex)));
                    }
                    else
                    {
                        auto methodConstName = builder.create<mlir_ts::SymbolRefOp>(
                            location, methodOrField.methodInfo.funcOp.getType(),
                            mlir::FlatSymbolRefAttr::get(builder.getContext(), methodOrField.methodInfo.funcOp.sym_name()));

                        vtableValue =
                            builder.create<mlir_ts::InsertPropertyOp>(location, virtTuple, methodConstName, vtableValue,
                                                                      builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex)));
                    }

                    fieldIndex++;
                }

                return std::pair<mlir::Type, mlir::Value>{virtTuple, vtableValue};
            },
            genContext);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassBaseInterfaces(mlir::Location location, ClassInfo::TypePtr newClassPtr, const GenContext &genContext)
    {
        for (auto &baseClass : newClassPtr->baseClasses)
        {
            for (auto &implement : baseClass->implements)
            {
                if (implement.processed)
                {
                    continue;
                }

                if (mlir::failed(mlirGenClassVirtualTableDefinitionForInterface(location, newClassPtr, implement.interface, genContext)))
                {
                    return mlir::failure();
                }

                implement.processed = true;
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassHeritageClauseImplements(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                             HeritageClause heritageClause, const GenContext &genContext)
    {
        if (heritageClause->token != SyntaxKind::ImplementsKeyword)
        {
            return mlir::success();
        }

        for (auto &implementingType : heritageClause->types)
        {
            if (implementingType->processed)
            {
                continue;
            }

            auto ifaceType = mlirGen(implementingType, genContext);
            auto success = false;
            TypeSwitch<mlir::Type>(ifaceType.getType())
                .template Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                    auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                    success = !failed(
                        mlirGenClassVirtualTableDefinitionForInterface(loc(implementingType), newClassPtr, interfaceInfo, genContext));
                })
                .Default([&](auto type) { llvm_unreachable("not implemented"); });

            if (!success)
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::Type getVirtualTableType(llvm::SmallVector<VirtualMethodOrFieldInfo> &virtualTable)
    {
        MLIRCodeLogic mcl(builder);

        llvm::SmallVector<mlir_ts::FieldInfo> fields;
        for (auto vtableRecord : virtualTable)
        {
            if (vtableRecord.isField)
            {
                fields.push_back({vtableRecord.fieldInfo.id, mlir_ts::RefType::get(vtableRecord.fieldInfo.type)});
            }
            else
            {
                fields.push_back({mcl.TupleFieldName(vtableRecord.methodInfo.name), vtableRecord.methodInfo.funcOp.getType()});
            }
        }

        auto virtTuple = getTupleType(fields);
        return virtTuple;
    }

    mlir::Type getVirtualTableType(llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> &virtualTable)
    {
        MLIRCodeLogic mcl(builder);

        llvm::SmallVector<mlir_ts::FieldInfo> fields;
        for (auto vtableRecord : virtualTable)
        {
            if (vtableRecord.isInterfaceVTable)
            {
                fields.push_back({mcl.TupleFieldName(vtableRecord.methodInfo.name), getOpaqueType()});
            }
            else
            {
                fields.push_back({mcl.TupleFieldName(vtableRecord.methodInfo.name), vtableRecord.methodInfo.funcOp.getType()});
            }
        }

        auto virtTuple = getTupleType(fields);
        return virtTuple;
    }

    mlir::LogicalResult mlirGenClassVirtualTableDefinition(mlir::Location location, ClassInfo::TypePtr newClassPtr,
                                                           const GenContext &genContext)
    {
        if (!newClassPtr->getHasVirtualTable() || newClassPtr->isAbstract)
        {
            return mlir::success();
        }

        // TODO: ...
        llvm::SmallVector<VirtualMethodOrInterfaceVTableInfo> virtualTable;
        newClassPtr->getVirtualTable(virtualTable);

        MLIRTypeHelper mth(builder.getContext());

        // register global
        auto fullClassVTableFieldName = concat(newClassPtr->fullName, VTABLE_NAME);
        registerVariable(
            location, fullClassVTableFieldName, true, VariableClass::Var,
            [&]() {
                // build vtable from names of methods

                MLIRCodeLogic mcl(builder);

                auto virtTuple = getVirtualTableType(virtualTable);

                mlir::Value vtableValue = builder.create<mlir_ts::UndefOp>(location, virtTuple);
                auto fieldIndex = 0;
                for (auto vtRecord : virtualTable)
                {
                    if (vtRecord.isInterfaceVTable)
                    {
                        // TODO: write correct full name for vtable
                        auto fullClassInterfaceVTableFieldName = concat(newClassPtr->fullName, vtRecord.methodInfo.name, VTABLE_NAME);
                        auto interfaceVTableValue =
                            resolveFullNameIdentifier(location, fullClassInterfaceVTableFieldName, true, genContext);

                        assert(interfaceVTableValue);

                        auto interfaceVTableValueAsAny = cast(location, getOpaqueType(), interfaceVTableValue, genContext);

                        vtableValue =
                            builder.create<mlir_ts::InsertPropertyOp>(location, virtTuple, interfaceVTableValueAsAny, vtableValue,
                                                                      builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex++)));
                    }
                    else
                    {
                        auto methodConstName = builder.create<mlir_ts::SymbolRefOp>(
                            location, vtRecord.methodInfo.funcOp.getType(),
                            mlir::FlatSymbolRefAttr::get(builder.getContext(), vtRecord.methodInfo.funcOp.sym_name()));

                        vtableValue =
                            builder.create<mlir_ts::InsertPropertyOp>(location, virtTuple, methodConstName, vtableValue,
                                                                      builder.getArrayAttr(mth.getStructIndexAttrValue(fieldIndex++)));
                    }
                }

                return std::pair<mlir::Type, mlir::Value>{virtTuple, vtableValue};
            },
            genContext);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenClassMethodMember(ClassLikeDeclaration classDeclarationAST, ClassInfo::TypePtr newClassPtr,
                                                 ClassElement classMember, const GenContext &genContext)
    {
        if (classMember->processed)
        {
            return mlir::success();
        }

        auto location = loc(classMember);

        auto &methodInfos = newClassPtr->methods;

        mlir::Value initValue;
        mlir::Attribute fieldId;
        mlir::Type type;
        StringRef memberNamePtr;

        auto isConstructor = classMember == SyntaxKind::Constructor;
        auto isStatic = hasModifier(classMember, SyntaxKind::StaticKeyword);
        auto isAbstract = hasModifier(classMember, SyntaxKind::AbstractKeyword);
        auto isVirtual = (classMember->transformFlags & TransformFlags::ForceVirtual) == TransformFlags::ForceVirtual;
#ifdef ALL_METHODS_VIRTUAL
        isVirtual = !isConstructor;
#endif
        if (classMember == SyntaxKind::MethodDeclaration || isConstructor || classMember == SyntaxKind::GetAccessor ||
            classMember == SyntaxKind::SetAccessor)
        {
            auto funcLikeDeclaration = classMember.as<FunctionLikeDeclarationBase>();
            std::string methodName;
            std::string propertyName;
            getMethodNameOrPropertyName(funcLikeDeclaration, methodName, propertyName);

            if (methodName.empty())
            {
                llvm_unreachable("not implemented");
                return mlir::failure();
            }

            classMember->parent = classDeclarationAST;

            auto funcGenContext = GenContext(genContext);
            funcGenContext.thisType = newClassPtr->classType;
            funcGenContext.passResult = nullptr;
            if (isConstructor)
            {
                if (isStatic && !genContext.allowPartialResolve)
                {
                    auto parentModule = theModule;

                    MLIRCodeLogicHelper mclh(builder, location);

                    builder.setInsertionPointToStart(parentModule.getBody());
                    mclh.seekLast(parentModule.getBody());

                    auto funcName = getNameOfFunction(classMember, genContext);

                    builder.create<mlir_ts::GlobalConstructorOp>(location, StringRef(std::get<0>(funcName)));
                }

                // adding missing statements
                generateConstructorStatements(classDeclarationAST, isStatic, funcGenContext);
            }

            auto res = mlirGenFunctionLikeDeclaration(funcLikeDeclaration, funcGenContext);
            if (mlir::failed(std::get<0>(res)))
            {
                return mlir::failure();
            }

            auto funcOp = std::get<1>(res);

            funcLikeDeclaration->processed = true;

            if (newClassPtr->getMethodIndex(methodName) < 0)
            {
                methodInfos.push_back({methodName, funcOp.getType(), funcOp, isStatic, isAbstract || isVirtual, -1});
            }

            if (propertyName.size() > 0)
            {
                addAccessor(newClassPtr, classMember, propertyName, funcOp, isStatic, isAbstract || isVirtual);
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult generateConstructorStatements(ClassLikeDeclaration classDeclarationAST, bool staticConstructor,
                                                      const GenContext &genContext)
    {
        NodeFactory nf(NodeFactoryFlags::None);

        for (auto &classMember : classDeclarationAST->members)
        {
            auto isStatic = hasModifier(classMember, SyntaxKind::StaticKeyword);
            if (classMember == SyntaxKind::PropertyDeclaration)
            {
                if (isStatic != staticConstructor)
                {
                    continue;
                }

                auto propertyDeclaration = classMember.as<PropertyDeclaration>();
                if (!propertyDeclaration->initializer)
                {
                    continue;
                }

                if (staticConstructor)
                {
                    auto isConst = false;
                    evaluate(
                        propertyDeclaration->initializer, [&](mlir::Value val) { isConst = isConstValue(val); }, genContext);
                    if (isConst)
                    {
                        continue;
                    }
                }

                auto memberNamePtr = MLIRHelper::getName(propertyDeclaration->name, stringAllocator);
                if (memberNamePtr.empty())
                {
                    llvm_unreachable("not implemented");
                    return mlir::failure();
                }

                auto _this = nf.createIdentifier(S(THIS_NAME));
                auto _name = nf.createIdentifier(stows(std::string(memberNamePtr)));
                auto _this_name = nf.createPropertyAccessExpression(_this, _name);
                auto _this_name_equal =
                    nf.createBinaryExpression(_this_name, nf.createToken(SyntaxKind::EqualsToken), propertyDeclaration->initializer);
                auto expr_statement = nf.createExpressionStatement(_this_name_equal);

                const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());
            }

            if (classMember == SyntaxKind::Constructor)
            {
                if (isStatic != staticConstructor)
                {
                    continue;
                }

                auto constructorDeclaration = classMember.as<ConstructorDeclaration>();
                for (auto &parameter : constructorDeclaration->parameters)
                {
                    auto isPublic = hasModifier(parameter, SyntaxKind::PublicKeyword);
                    auto isProtected = hasModifier(parameter, SyntaxKind::ProtectedKeyword);
                    auto isPrivate = hasModifier(parameter, SyntaxKind::PrivateKeyword);

                    if (!(isPublic || isProtected || isPrivate))
                    {
                        continue;
                    }

                    auto propertyNamePtr = MLIRHelper::getName(parameter->name, stringAllocator);
                    if (propertyNamePtr.empty())
                    {
                        llvm_unreachable("not implemented");
                        return mlir::failure();
                    }

                    auto _this = nf.createIdentifier(stows(THIS_NAME));
                    auto _name = nf.createIdentifier(stows(std::string(propertyNamePtr)));
                    auto _this_name = nf.createPropertyAccessExpression(_this, _name);
                    auto _this_name_equal = nf.createBinaryExpression(_this_name, nf.createToken(SyntaxKind::EqualsToken), _name);
                    auto expr_statement = nf.createExpressionStatement(_this_name_equal);

                    const_cast<GenContext &>(genContext).generatedStatements.push_back(expr_statement.as<Statement>());
                }
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult registerGenericInterface(InterfaceDeclaration interfaceDeclarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(interfaceDeclarationAST->name);
        if (!name.empty())
        {
            auto namePtr = StringRef(name).copy(stringAllocator);
            auto fullNamePtr = getFullNamespaceName(namePtr);
            if (fullNameGenericInterfacesMap.count(fullNamePtr))
            {
                return mlir::success();
            }

            llvm::SmallVector<TypeParameterDOM::TypePtr> typeParameters;
            if (mlir::failed(processTypeParameters(interfaceDeclarationAST->typeParameters, typeParameters, genContext)))
            {
                return mlir::failure();
            }

            GenericInterfaceInfo::TypePtr newGenericInterfacePtr = std::make_shared<GenericInterfaceInfo>();
            newGenericInterfacePtr->name = namePtr;
            newGenericInterfacePtr->fullName = fullNamePtr;
            newGenericInterfacePtr->typeParams = typeParameters;
            newGenericInterfacePtr->interfaceDeclaration = interfaceDeclarationAST;

            mlirGenInterfaceType(newGenericInterfacePtr);

            getGenericInterfacesMap().insert({namePtr, newGenericInterfacePtr});
            fullNameGenericInterfacesMap.insert(fullNamePtr, newGenericInterfacePtr);

            return mlir::success();
        }

        return mlir::failure();
    }

    template <typename T> std::string getNameWithArguments(T declarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(declarationAST->name);
        if (genContext.typeParamsWithArgs.size() && declarationAST->typeParameters.size())
        {
            name.append("<");
            auto next = false;
            for (auto typeParam : declarationAST->typeParameters)
            {
                if (next)
                {
                    name.append(",");
                }

                auto type = getType(typeParam, genContext);
                llvm::raw_string_ostream s(name);
                s << type;
                next = false;
            }

            name.append(">");
        }

        return name;
    }

    std::string getSpecializedInterfaceName(GenericInterfaceInfo::TypePtr geneticInterfacePtr, const GenContext &genContext)
    {
        auto name = geneticInterfacePtr->fullName.str();
        if (genContext.typeParamsWithArgs.size())
        {
            name.append("<");
            auto next = false;
            for (auto typeParam : geneticInterfacePtr->typeParams)
            {
                if (next)
                {
                    name.append(",");
                }

                auto type = getResolveTypeParameter(typeParam->getName(), genContext);
                llvm::raw_string_ostream s(name);
                s << type;
                next = false;
            }

            name.append(">");
        }

        return name;
    }

    mlir_ts::InterfaceType getSpecializationInterfaceType(GenericInterfaceInfo::TypePtr geneticInterfacePtr, const GenContext &genContext)
    {
        auto fullSpecializedInterfaceName = getSpecializedInterfaceName(geneticInterfacePtr, genContext);
        auto interfaceInfoType = getInterfaceInfoByFullName(fullSpecializedInterfaceName);
        return interfaceInfoType->interfaceType;
    }

    InterfaceInfo::TypePtr mlirGenInterfaceInfo(InterfaceDeclaration interfaceDeclarationAST, bool &declareInterface,
                                                const GenContext &genContext)
    {
        auto name = getNameWithArguments(interfaceDeclarationAST, genContext);
        return mlirGenInterfaceInfo(name, declareInterface, genContext);
    }

    InterfaceInfo::TypePtr mlirGenInterfaceInfo(std::string name, bool &declareInterface, const GenContext &genContext)
    {
        declareInterface = false;

        auto namePtr = StringRef(name).copy(stringAllocator);
        auto fullNamePtr = getFullNamespaceName(namePtr);

        InterfaceInfo::TypePtr newInterfacePtr;
        if (fullNameInterfacesMap.count(fullNamePtr))
        {
            newInterfacePtr = fullNameInterfacesMap.lookup(fullNamePtr);
            getInterfacesMap().insert({namePtr, newInterfacePtr});
            declareInterface = !newInterfacePtr->interfaceType;
        }
        else
        {
            // register class
            newInterfacePtr = std::make_shared<InterfaceInfo>();
            newInterfacePtr->name = namePtr;
            newInterfacePtr->fullName = fullNamePtr;

            getInterfacesMap().insert({namePtr, newInterfacePtr});
            fullNameInterfacesMap.insert(fullNamePtr, newInterfacePtr);
            declareInterface = true;
        }

        if (declareInterface)
        {
            mlirGenInterfaceType(newInterfacePtr);
        }

        return newInterfacePtr;
    }

    mlir::LogicalResult mlirGenInterfaceHeritageClauseExtends(InterfaceDeclaration interfaceDeclarationAST,
                                                              InterfaceInfo::TypePtr newInterfacePtr, HeritageClause heritageClause,
                                                              bool declareClass, const GenContext &genContext)
    {
        if (heritageClause->token != SyntaxKind::ExtendsKeyword)
        {
            return mlir::success();
        }

        for (auto &extendsType : heritageClause->types)
        {
            if (extendsType->processed)
            {
                continue;
            }

            auto ifaceType = mlirGen(extendsType, genContext);
            auto success = false;
            TypeSwitch<mlir::Type>(ifaceType.getType())
                .template Case<mlir_ts::InterfaceType>([&](auto interfaceType) {
                    auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                    newInterfacePtr->extends.push_back({-1, interfaceInfo});
                    success = true;
                    extendsType->processed = true;
                })
                .Default([&](auto type) { llvm_unreachable("not implemented"); });

            if (!success)
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(InterfaceDeclaration interfaceDeclarationAST, const GenContext &genContext)
    {
        // do not proceed for Generic Interfaces for declaration
        if (interfaceDeclarationAST->typeParameters.size() > 0 && genContext.typeParamsWithArgs.size() == 0)
        {
            return registerGenericInterface(interfaceDeclarationAST, genContext);
        }

        auto declareInterface = false;
        auto newInterfacePtr = mlirGenInterfaceInfo(interfaceDeclarationAST, declareInterface, genContext);
        if (!newInterfacePtr)
        {
            return mlir::failure();
        }

        // do not process specialized interface second time;
        if (!declareInterface && interfaceDeclarationAST->typeParameters.size() > 0 && genContext.typeParamsWithArgs.size() > 0)
        {
            return mlir::success();
        }

        auto location = loc(interfaceDeclarationAST);

        auto ifaceGenContext = GenContext(genContext);
        ifaceGenContext.thisType = newInterfacePtr->interfaceType;

        for (auto &heritageClause : interfaceDeclarationAST->heritageClauses)
        {
            if (mlir::failed(mlirGenInterfaceHeritageClauseExtends(interfaceDeclarationAST, newInterfacePtr, heritageClause,
                                                                   declareInterface, genContext)))
            {
                return mlir::failure();
            }
        }

        newInterfacePtr->recalcOffsets();

        // clear all flags
        for (auto &interfaceMember : interfaceDeclarationAST->members)
        {
            interfaceMember->processed = false;
        }

        // add methods when we have classType
        auto notResolved = 0;
        do
        {
            auto lastTimeNotResolved = notResolved;
            notResolved = 0;

            for (auto &interfaceMember : interfaceDeclarationAST->members)
            {
                if (mlir::failed(mlirGenInterfaceMethodMember(interfaceDeclarationAST, newInterfacePtr, interfaceMember, declareInterface,
                                                              ifaceGenContext)))
                {
                    notResolved++;
                }
            }

            // repeat if not all resolved
            if (lastTimeNotResolved > 0 && lastTimeNotResolved == notResolved)
            {
                // interface can depend on other interface declarations
                // theModule.emitError("can't resolve dependencies in intrerface: ") << newInterfacePtr->name;
                return mlir::failure();
            }

        } while (notResolved > 0);

        return mlir::success();
    }

    template <typename T> mlir::LogicalResult mlirGenInterfaceType(T newInterfacePtr)
    {
        if (newInterfacePtr)
        {
            newInterfacePtr->interfaceType = getInterfaceType(newInterfacePtr->fullName);
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGenInterfaceMethodMember(InterfaceDeclaration interfaceDeclarationAST, InterfaceInfo::TypePtr newInterfacePtr,
                                                     TypeElement interfaceMember, bool declareInterface, const GenContext &genContext)
    {
        if (interfaceMember->processed)
        {
            return mlir::success();
        }

        auto location = loc(interfaceMember);

        auto &fieldInfos = newInterfacePtr->fields;
        auto &methodInfos = newInterfacePtr->methods;

        mlir::Value initValue;
        mlir::Attribute fieldId;
        mlir::Type type;
        StringRef memberNamePtr;

        MLIRCodeLogic mcl(builder);

        if (interfaceMember == SyntaxKind::PropertySignature)
        {
            // property declaration
            auto propertySignature = interfaceMember.as<PropertySignature>();
            auto isConditional = !!propertySignature->questionToken;

            auto memberNamePtr = MLIRHelper::getName(propertySignature->name, stringAllocator);
            if (memberNamePtr.empty())
            {
                llvm_unreachable("not implemented");
                return mlir::failure();
            }

            fieldId = mcl.TupleFieldName(memberNamePtr);

            auto typeAndInit = getTypeAndInit(propertySignature, genContext);
            type = typeAndInit.first;

            // fix type for fields with FuncType
            if (auto hybridFuncType = type.dyn_cast<mlir_ts::HybridFunctionType>())
            {
                MLIRTypeHelper mth(builder.getContext());
                auto funcType = getFunctionType(hybridFuncType.getInputs(), hybridFuncType.getResults());
                type = mth.getFunctionTypeAddingFirstArgType(funcType, getOpaqueType());
            }
            else if (auto funcType = type.dyn_cast<mlir_ts::FunctionType>())
            {
                MLIRTypeHelper mth(builder.getContext());
                type = mth.getFunctionTypeAddingFirstArgType(funcType, getOpaqueType());
            }

            LLVM_DEBUG(dbgs() << "\n!! interface field: " << fieldId << " type: " << type << "\n\n");

            if (isNoneType(type))
            {
                return mlir::failure();
            }

            if (declareInterface || newInterfacePtr->getFieldIndex(fieldId) == -1)
            {
                fieldInfos.push_back({fieldId, type, isConditional, newInterfacePtr->getNextVTableMemberIndex()});
            }
        }

        if (interfaceMember == SyntaxKind::MethodSignature)
        {
            auto methodSignature = interfaceMember.as<MethodSignature>();
            auto isConditional = !!methodSignature->questionToken;

            std::string methodName;
            std::string propertyName;
            getMethodNameOrPropertyName(methodSignature, methodName, propertyName);

            if (methodName.empty())
            {
                llvm_unreachable("not implemented");
                return mlir::failure();
            }

            interfaceMember->parent = interfaceDeclarationAST;

            auto funcGenContext = GenContext(genContext);
            funcGenContext.thisType = newInterfacePtr->interfaceType;
            funcGenContext.passResult = nullptr;

            auto res = mlirGenFunctionSignaturePrototype(methodSignature, true, funcGenContext);
            auto funcType = std::get<1>(res);

            if (!funcType)
            {
                return mlir::failure();
            }

            methodSignature->processed = true;

            if (declareInterface || newInterfacePtr->getMethodIndex(methodName) == -1)
            {
                methodInfos.push_back({methodName, funcType, isConditional, newInterfacePtr->getNextVTableMemberIndex()});
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult getMethodNameOrPropertyName(SignatureDeclarationBase methodSignature, std::string &methodName,
                                                    std::string &propertyName)
    {
        if (methodSignature == SyntaxKind::Constructor)
        {
            auto isStatic = hasModifier(methodSignature, SyntaxKind::StaticKeyword);
            if (isStatic)
            {
                methodName = std::string(STATIC_CONSTRUCTOR_NAME);
            }
            else
            {
                methodName = std::string(CONSTRUCTOR_NAME);
            }
        }
        else if (methodSignature == SyntaxKind::GetAccessor)
        {
            propertyName = MLIRHelper::getName(methodSignature->name);
            methodName = std::string("get_") + propertyName;
        }
        else if (methodSignature == SyntaxKind::SetAccessor)
        {
            propertyName = MLIRHelper::getName(methodSignature->name);
            methodName = std::string("set_") + propertyName;
        }
        else
        {
            methodName = MLIRHelper::getName(methodSignature->name);
        }

        return mlir::success();
    }

    void addAccessor(ClassInfo::TypePtr newClassPtr, ClassElement classMember, std::string &propertyName, mlir_ts::FuncOp funcOp,
                     bool isStatic, bool isVirtual)
    {
        auto &accessorInfos = newClassPtr->accessors;

        auto accessorIndex = newClassPtr->getAccessorIndex(propertyName);
        if (accessorIndex < 0)
        {
            accessorInfos.push_back({propertyName, {}, {}, isStatic, isVirtual});
            accessorIndex = newClassPtr->getAccessorIndex(propertyName);
        }

        assert(accessorIndex >= 0);

        if (classMember == SyntaxKind::GetAccessor)
        {
            newClassPtr->accessors[accessorIndex].get = funcOp;
        }
        else if (classMember == SyntaxKind::SetAccessor)
        {
            newClassPtr->accessors[accessorIndex].set = funcOp;
        }
    }

    mlir::Type evaluate(Expression expr, const GenContext &genContext)
    {
        // we need to add temporary block
        mlir::Type result;
        evaluate(
            expr, [&](mlir::Value val) { result = val.getType(); }, genContext);

        return result;
    }

    void evaluate(Expression expr, std::function<void(mlir::Value)> func, const GenContext &genContext)
    {
        if (!expr)
        {
            return;
        }

        // we need to add temporary block
        auto tempFuncType = mlir::FunctionType::get(builder.getContext(), ArrayRef<mlir::Type>(), ArrayRef<mlir::Type>());
        auto tempFuncOp = mlir::FuncOp::create(loc(expr), ".tempfunc", tempFuncType);
        auto &entryBlock = *tempFuncOp.addEntryBlock();

        {
            mlir::OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(&entryBlock);

            GenContext evalGenContext(genContext);
            evalGenContext.allowPartialResolve = true;
            auto initValue = mlirGen(expr, evalGenContext);
            if (initValue)
            {
                func(initValue);
            }
        }

        entryBlock.dropAllDefinedValueUses();
        entryBlock.dropAllUses();
        entryBlock.dropAllReferences();
        entryBlock.erase();

        tempFuncOp.erase();
    }

    mlir::Type evaluateProperty(mlir::Value exprValue, std::string propertyName, const GenContext &genContext)
    {
        auto location = exprValue.getLoc();
        // we need to add temporary block
        auto tempFuncType = mlir::FunctionType::get(builder.getContext(), llvm::None, llvm::None);
        auto tempFuncOp = mlir::FuncOp::create(location, ".tempfunc", tempFuncType);
        auto &entryBlock = *tempFuncOp.addEntryBlock();

        auto insertPoint = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(&entryBlock);

        mlir::Type result;
        GenContext evalGenContext(genContext);
        evalGenContext.allowPartialResolve = true;
        auto initValue = mlirGenPropertyAccessExpression(location, exprValue, propertyName, evalGenContext);
        if (initValue)
        {
            result = initValue.getType();
        }

        // remove temp block
        builder.restoreInsertionPoint(insertPoint);
        entryBlock.erase();
        tempFuncOp.erase();

        return result;
    }

    mlir::Value cast(mlir::Location location, mlir::Type type, mlir::Value value, const GenContext &genContext)
    {
        if (type == value.getType())
        {
            return value;
        }

        // class to string
        if (auto stringType = type.dyn_cast_or_null<mlir_ts::StringType>())
        {
            if (auto classType = value.getType().dyn_cast_or_null<mlir_ts::ClassType>())
            {
                return mlirGenCallThisMethod(location, value, "toString", undefined, undefined, genContext);
            }
        }

        // class to interface
        if (auto interfaceType = type.dyn_cast_or_null<mlir_ts::InterfaceType>())
        {
            if (auto classType = value.getType().dyn_cast_or_null<mlir_ts::ClassType>())
            {
                auto vtableAccess = mlirGenPropertyAccessExpression(location, value, VTABLE_NAME, genContext);

                auto classInfo = getClassInfoByFullName(classType.getName().getValue());
                assert(classInfo);

                auto implementIndex = classInfo->getImplementIndex(interfaceType.getName().getValue());
                if (implementIndex >= 0)
                {
                    auto interfaceVirtTableIndex = classInfo->implements[implementIndex].virtualIndex;

                    assert(genContext.allowPartialResolve || interfaceVirtTableIndex >= 0);

                    MLIRTypeHelper mth(builder.getContext());

                    auto interfaceVTablePtr = builder.create<mlir_ts::VTableOffsetRefOp>(
                        location, mth.getInterfaceVTableType(interfaceType), vtableAccess, interfaceVirtTableIndex);

                    auto newInterface =
                        builder.create<mlir_ts::NewInterfaceOp>(location, mlir::TypeRange{interfaceType}, value, interfaceVTablePtr);
                    return newInterface;
                }

                // create interface vtable from current class
                auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
                assert(interfaceInfo);

                if (auto createdInterfaceVTableForClass =
                        mlirGenCreateInterfaceVTableForClass(location, classInfo, interfaceInfo, genContext))
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!!"
                                            << "@ created interface:" << createdInterfaceVTableForClass << "\n";);
                    auto newInterface = builder.create<mlir_ts::NewInterfaceOp>(location, mlir::TypeRange{interfaceType}, value,
                                                                                createdInterfaceVTableForClass);

                    return newInterface;
                }

                emitError(location) << "type: " << classType << " missing interface: " << interfaceType;
                return mlir::Value();
            }
        }

        // tuple to interface
        if (auto interfaceType = type.dyn_cast_or_null<mlir_ts::InterfaceType>())
        {
            if (auto constTupleType = value.getType().dyn_cast_or_null<mlir_ts::ConstTupleType>())
            {
                return castTupleToInterface(location, value, constTupleType, interfaceType, genContext);
            }

            if (auto tupleType = value.getType().dyn_cast_or_null<mlir_ts::TupleType>())
            {
                return castTupleToInterface(location, value, tupleType, interfaceType, genContext);
            }
        }

        if (auto unionType = type.dyn_cast_or_null<mlir_ts::UnionType>())
        {
            MLIRTypeHelper mth(builder.getContext());
            mlir::Type baseType;
            if (!mth.isUnionTypeNeedsTag(unionType, baseType))
            {
                auto valueCasted = cast(location, baseType, value, genContext);
                VALIDATE(valueCasted, location)
                return builder.create<mlir_ts::CastOp>(location, type, valueCasted);
            }
        }

        // optional with union & interface inside
        if (auto optType = type.dyn_cast_or_null<mlir_ts::OptionalType>())
        {
            auto valueCasted = cast(location, optType.getElementType(), value, genContext);
            VALIDATE(valueCasted, location)
            return builder.create<mlir_ts::CreateOptionalOp>(location, optType, valueCasted);
        }

        return builder.create<mlir_ts::CastOp>(location, type, value);
    }

    mlir::Value castTupleToInterface(mlir::Location location, mlir::Value in, mlir::Type tupleTypeIn, mlir_ts::InterfaceType interfaceType,
                                     const GenContext &genContext)
    {
        MLIRTypeHelper mth(builder.getContext());

        auto tupleType = mth.convertConstTupleTypeToTupleType(tupleTypeIn);
        auto interfaceInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());

        auto inEffective = in;

        if (mlir::failed(canCastTupleToInterface(tupleType.cast<mlir_ts::TupleType>(), interfaceInfo)))
        {
            MLIRTypeHelper mth(builder.getContext());
            SmallVector<mlir_ts::FieldInfo> fields;
            if (mlir::succeeded(interfaceInfo->getTupleTypeFields(fields, mth)))
            {
                auto newInterfaceTupleType = getTupleType(fields);
                inEffective = cast(location, newInterfaceTupleType, inEffective, genContext);
                if (inEffective)
                {
                    tupleType = newInterfaceTupleType;
                }
                else
                {
                    return mlir::Value();
                }
            }
            else
            {
                return mlir::Value();
            }
        }

        // TODO: finish it
        // convert Tuple to Object
        auto objType = mlir_ts::ObjectType::get(tupleType);

        auto valueAddr = builder.create<mlir_ts::NewOp>(location, mlir_ts::ValueRefType::get(tupleType), builder.getBoolAttr(false));
        builder.create<mlir_ts::StoreOp>(location, inEffective, valueAddr);
        auto inCasted = builder.create<mlir_ts::CastOp>(location, objType, valueAddr);

        if (auto createdInterfaceVTableForObject = mlirGenCreateInterfaceVTableForObject(location, objType, interfaceInfo, genContext))
        {

            LLVM_DEBUG(llvm::dbgs() << "\n!!"
                                    << "@ created interface:" << createdInterfaceVTableForObject << "\n";);
            auto newInterface = builder.create<mlir_ts::NewInterfaceOp>(location, mlir::TypeRange{interfaceType}, inCasted,
                                                                        createdInterfaceVTableForObject);

            return newInterface;
        }

        return mlir::Value();
    }

    mlir::Type getType(Node typeReferenceAST, const GenContext &genContext)
    {
        auto kind = (SyntaxKind)typeReferenceAST;
        if (kind == SyntaxKind::BooleanKeyword)
        {
            return getBooleanType();
        }
        else if (kind == SyntaxKind::NumberKeyword)
        {
            return getNumberType();
        }
        else if (kind == SyntaxKind::BigIntKeyword)
        {
            return getBigIntType();
        }
        else if (kind == SyntaxKind::StringKeyword)
        {
            return getStringType();
        }
        else if (kind == SyntaxKind::VoidKeyword)
        {
            return getVoidType();
        }
        else if (kind == SyntaxKind::FunctionType)
        {
            return getFunctionType(typeReferenceAST.as<FunctionTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::ConstructorType)
        {
            // TODO: do I need to add flag to FunctionType to show that this is ConstructorType?
            return getConstructorType(typeReferenceAST.as<ConstructorTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::CallSignature)
        {
            return getCallSignature(typeReferenceAST.as<CallSignatureDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::MethodSignature)
        {
            return getMethodSignature(typeReferenceAST.as<MethodSignature>(), genContext);
        }
        else if (kind == SyntaxKind::ConstructSignature)
        {
            return getConstructSignature(typeReferenceAST.as<ConstructSignatureDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::IndexSignature)
        {
            return getIndexSignature(typeReferenceAST.as<IndexSignatureDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::TupleType)
        {
            return getTupleType(typeReferenceAST.as<TupleTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeLiteral)
        {
            // TODO: review it, I think it should be ObjectType
            // return getObjectType(getTupleType(typeReferenceAST.as<TypeLiteralNode>(), genContext));
            return getTupleType(typeReferenceAST.as<TypeLiteralNode>(), genContext);
        }
        else if (kind == SyntaxKind::ArrayType)
        {
            return getArrayType(typeReferenceAST.as<ArrayTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::UnionType)
        {
            return getUnionType(typeReferenceAST.as<UnionTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::IntersectionType)
        {
            return getIntersectionType(typeReferenceAST.as<IntersectionTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::ParenthesizedType)
        {
            return getParenthesizedType(typeReferenceAST.as<ParenthesizedTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::LiteralType)
        {
            return getLiteralType(typeReferenceAST.as<LiteralTypeNode>());
        }
        else if (kind == SyntaxKind::TypeReference)
        {
            return getTypeByTypeReference(typeReferenceAST.as<TypeReferenceNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeQuery)
        {
            return getTypeByTypeQuery(typeReferenceAST.as<TypeQueryNode>(), genContext);
        }
        else if (kind == SyntaxKind::ObjectKeyword)
        {
            return getObjectType(getAnyType());
        }
        else if (kind == SyntaxKind::AnyKeyword)
        {
            return getAnyType();
        }
        else if (kind == SyntaxKind::UnknownKeyword)
        {
            // TODO: do I need to have special type?
            return getUnknownType();
        }
        else if (kind == SyntaxKind::SymbolKeyword)
        {
            return getSymbolType();
        }
        else if (kind == SyntaxKind::UndefinedKeyword)
        {
            return getUndefinedType();
        }
        else if (kind == SyntaxKind::UndefinedKeyword)
        {
            return getUndefinedType();
        }
        else if (kind == SyntaxKind::TypePredicate)
        {
            // in runtime it is boolean (it is needed to track types)
            return getBooleanType();
        }
        else if (kind == SyntaxKind::ThisType)
        {
            assert(genContext.thisType);
            return genContext.thisType;
        }
        else if (kind == SyntaxKind::Unknown)
        {
            // in case type is not provided
            return getAnyType();
        }
        else if (kind == SyntaxKind::ConditionalType)
        {
            return getConditionalType(typeReferenceAST.as<ConditionalTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeOperator)
        {
            return getTypeOperator(typeReferenceAST.as<TypeOperatorNode>(), genContext);
        }
        else if (kind == SyntaxKind::IndexedAccessType)
        {
            return getIndexedAccessType(typeReferenceAST.as<IndexedAccessTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::MappedType)
        {
            return getMappedType(typeReferenceAST.as<MappedTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TemplateLiteralType)
        {
            return getTemplateLiteralType(typeReferenceAST.as<TemplateLiteralTypeNode>(), genContext);
        }
        else if (kind == SyntaxKind::TypeParameter)
        {
            return getResolveTypeParameter(typeReferenceAST.as<TypeParameterDeclaration>(), genContext);
        }

        llvm_unreachable("not implemented type declaration");
        // return getAnyType();
    }

    mlir::Type getResolveTypeParameter(StringRef typeParamName, const GenContext &genContext)
    {
        auto found = genContext.typeParamsWithArgs.find(typeParamName);
        if (found != genContext.typeParamsWithArgs.end())
        {
            auto type = (*found).getValue().second;

            LLVM_DEBUG(llvm::dbgs() << "\n!! type gen. param [" << typeParamName << "] -> [" << type << "]\n";);

            return type;
        }

        return mlir::Type();
    }

    mlir::Type getResolveTypeParameter(TypeParameterDeclaration typeParameterDeclaration, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(typeParameterDeclaration->name);
        if (name.empty())
        {
            llvm_unreachable("not implemented");
            return mlir::Type();
        }

        return getResolveTypeParameter(name, genContext);
    }

    mlir::Type getTypeByTypeName(Node node, const GenContext &genContext)
    {
        mlir::Value value;
        if (node == SyntaxKind::QualifiedName)
        {
            value = mlirGen(node.as<QualifiedName>(), genContext);
        }
        else
        {
            value = mlirGen(node.as<Expression>(), genContext);
        }

        if (value)
        {
            auto type = value.getType();

            // extra code for extracting enum storage type
            // TODO: think if you can avoid doing it
            if (auto enumType = type.dyn_cast_or_null<mlir_ts::EnumType>())
            {
                return enumType.getElementType();
            }

            assert(type);

            return type;
        }

        llvm_unreachable("not implemented");
    }

    mlir::Type getFirstTypeFromTypeArguments(NodeArray<TypeNode> &typeArguments, const GenContext &genContext, bool extractType = false)
    {
        auto type = getType(typeArguments->front(), genContext);
        if (extractType)
        {
            if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
            {
                type = literalType.getElementType();
            }
        }

        return type;
    }

    mlir::Type getSecondTypeFromTypeArguments(NodeArray<TypeNode> &typeArguments, const GenContext &genContext, bool extractType = false)
    {
        auto type = getType(typeArguments[1], genContext);
        if (extractType)
        {
            if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
            {
                type = literalType.getElementType();
            }
        }

        return type;
    }

    mlir::LogicalResult zipTypeParameterWithArgument(mlir::Location location,
                                                     llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs,
                                                     const ts::TypeParameterDOM::TypePtr &typeParam, mlir::Type type)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! assigning generic type: " << typeParam->getName() << " type: " << type << "\n";);

        MLIRTypeHelper mth(builder.getContext());
        if (typeParam->getConstraint() && !mth.extendsType(type, typeParam->getConstraint()))
        {
            emitWarning(location, "") << "Type " << type << " does not satisfy the constraint " << typeParam->getConstraint() << ".";
            return mlir::failure();
        }

        pairs.insert({typeParam->getName(), std::make_pair(typeParam, type)});

        return mlir::success();
    }

    mlir::LogicalResult zipTypeParametersWithArguments(mlir::Location location, llvm::ArrayRef<TypeParameterDOM::TypePtr> typeParams,
                                                       NodeArray<TypeNode> typeArgs,
                                                       llvm::StringMap<std::pair<TypeParameterDOM::TypePtr, mlir::Type>> &pairs,
                                                       const GenContext &genContext)
    {
        for (auto index = 0; index < typeParams.size(); index++)
        {
            auto &typeParam = typeParams[index];
            auto type = getType(typeArgs[index], genContext);

            if (mlir::failed(zipTypeParameterWithArgument(location, pairs, typeParam, type)))
            {
                return mlir::failure();
            }
        }

        return mlir::success();
    }

    mlir::Type getTypeByTypeReference(TypeReferenceNode typeReferenceAST, const GenContext &genContext)
    {
        // check utility types
        if (typeReferenceAST->typeArguments->size() > 0)
        {
            auto name = MLIRHelper::getName(typeReferenceAST->typeName);

            // try to resolve from type alias first
            if (getGenericTypeAliasMap().count(name))
            {
                auto genericTypeAliasInfo = getGenericTypeAliasMap().lookup(name);
                GenContext genericTypeGenContext(genContext);

                auto typeParams = std::get<0>(genericTypeAliasInfo);
                auto typeNode = std::get<1>(genericTypeAliasInfo);

                if (mlir::failed(zipTypeParametersWithArguments(loc(typeReferenceAST), typeParams, typeReferenceAST->typeArguments,
                                                                genericTypeGenContext.typeParamsWithArgs, genContext)))
                {
                    return getNeverType();
                }

                auto newType = getType(typeNode, genericTypeGenContext);
                return newType;
            }

            if (getGenericClassesMap().count(name))
            {
                auto genericClassTypeInfo = getGenericClassesMap().lookup(name);
                auto classType = genericClassTypeInfo->classType;
                auto specType =
                    instantiateSpecializedClassType(loc(typeReferenceAST), classType, typeReferenceAST->typeArguments, genContext);
                return specType;
            }

            if (getGenericInterfacesMap().count(name))
            {
                auto genericInterfaceTypeInfo = getGenericInterfacesMap().lookup(name);
                auto interfaceType = genericInterfaceTypeInfo->interfaceType;
                auto specType =
                    instantiateSpecializedInterfaceType(loc(typeReferenceAST), interfaceType, typeReferenceAST->typeArguments, genContext);
                return specType;
            }

            // can be utility type
            if (name == "TypeOf")
            {
                return getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext, true);
            }

            if (name == "Readonly")
            {
                // TODO: ???
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return elemnentType;
            }

            if (name == "Partial")
            {
                // TODO: ???
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return elemnentType;
            }

            if (name == "Required")
            {
                // TODO: ???
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return elemnentType;
            }

            if (name == "NonNullable")
            {
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return NonNullableTypes(elemnentType);
            }

            if (name == "Exclude")
            {
                auto firstType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                auto secondType = getSecondTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return ExcludeTypes(firstType, secondType);
            }

            if (name == "Extract")
            {
                auto firstType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                auto secondType = getSecondTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return ExtractTypes(firstType, secondType);
            }

            if (name == "Array")
            {
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return getArrayType(elemnentType);
            }

            if (name == "ReadonlyArray")
            {
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return getArrayType(elemnentType);
            }

            if (name == "ReturnType")
            {
                auto elementType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                if (genContext.allowPartialResolve && !elementType)
                {
                    return mlir::Type();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! ReturnType Of: " << elementType;);
                auto retType = getReturnTypeFromFuncRef(elementType);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            }

            if (name == "Parameters")
            {
                auto elementType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                if (genContext.allowPartialResolve && !elementType)
                {
                    return mlir::Type();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << elementType;);
                auto retType = getParamsTupleTypeFromFuncRef(elementType);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            }

            if (name == "ConstructorParameters")
            {
                auto elementType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                if (genContext.allowPartialResolve && !elementType)
                {
                    return mlir::Type();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << elementType;);
                auto retType = getParamsTupleTypeFromFuncRef(elementType);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            }

            // TODO: add flag to FunctionType to see if type has this parameter (the same way as VarArg)
            if (name == "ThisParameterType")
            {
                auto elementType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                if (genContext.allowPartialResolve && !elementType)
                {
                    return mlir::Type();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << elementType;);
                auto retType = getFirstParamFromFuncRef(elementType);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            }

            if (name == "OmitThisParameter")
            {
                auto elementType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                if (genContext.allowPartialResolve && !elementType)
                {
                    return mlir::Type();
                }

                LLVM_DEBUG(llvm::dbgs() << "\n!! ElementType Of: " << elementType;);
                auto retType = getOmitThisFunctionTypeFromFuncRef(elementType);
                LLVM_DEBUG(llvm::dbgs() << " is " << retType << "\n";);
                return retType;
            }

            // ThisType is dummy type
            if (name == "ThisType")
            {
                auto elementType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return elementType;
            }

            if (name == "Awaited")
            {
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return elemnentType;
            }

            if (name == "Promise")
            {
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return elemnentType;
            }

            // string types
            if (name == "Uppercase")
            {
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return UppercaseType(elemnentType);
            }

            if (name == "Lowercase")
            {
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return LowercaseType(elemnentType);
            }

            if (name == "Capitalize")
            {
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return CapitalizeType(elemnentType);
            }

            if (name == "Uncapitalize")
            {
                auto elemnentType = getFirstTypeFromTypeArguments(typeReferenceAST->typeArguments, genContext);
                return UncapitalizeType(elemnentType);
            }
        }

        return getTypeByTypeName(typeReferenceAST->typeName, genContext);
    }

    mlir::Type StringLiteralTypeFunc(mlir::Type type, std::function<std::string(StringRef)> f)
    {
        if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
        {
            if (literalType.getElementType().isa<mlir_ts::StringType>())
            {
                auto newStr = f(literalType.getValue().cast<mlir::StringAttr>().getValue());
                auto copyVal = StringRef(newStr).copy(stringAllocator);
                return mlir_ts::LiteralType::get(builder.getStringAttr(copyVal), getStringType());
            }
        }

        return mlir::Type();
    }

    mlir::Type UppercaseType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type, [](auto val) { return val.upper(); });
    }

    mlir::Type LowercaseType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type, [](auto val) { return val.lower(); });
    }

    mlir::Type CapitalizeType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type, [](auto val) { return val.slice(0, 1).upper().append(val.slice(1, val.size())); });
    }

    mlir::Type UncapitalizeType(mlir::Type type)
    {
        return StringLiteralTypeFunc(type, [](auto val) { return val.slice(0, 1).lower().append(val.slice(1, val.size())); });
    }

    mlir::Type NonNullableTypes(mlir::Type type)
    {
        SmallPtrSet<mlir::Type, 2> types;

        MLIRHelper::LoadTypes(types, type);

        SmallVector<mlir::Type> resTypes;
        for (auto item : types)
        {
            if (item.isa<mlir_ts::NullType>() || item == getOptionalType(getUndefPlaceHolderType()))
            {
                continue;
            }

            resTypes.push_back(item);
        }

        return getUnionType(resTypes);
    }

    mlir::Type ExcludeTypes(mlir::Type type, mlir::Type exclude)
    {
        SmallPtrSet<mlir::Type, 2> types;
        SmallPtrSet<mlir::Type, 2> excludeTypes;

        MLIRHelper::LoadTypes(types, type);
        MLIRHelper::LoadTypes(excludeTypes, exclude);

        SmallVector<mlir::Type> resTypes;
        for (auto item : types)
        {
            if (excludeTypes.count(item))
            {
                continue;
            }

            resTypes.push_back(item);
        }

        return getUnionType(resTypes);
    }

    mlir::Type ExtractTypes(mlir::Type type, mlir::Type extract)
    {
        SmallPtrSet<mlir::Type, 2> types;
        SmallPtrSet<mlir::Type, 2> extractTypes;

        MLIRHelper::LoadTypes(types, type);
        MLIRHelper::LoadTypes(extractTypes, extract);

        SmallVector<mlir::Type> resTypes;
        for (auto item : types)
        {
            if (extractTypes.count(item))
            {
                resTypes.push_back(item);
            }
        }

        return getUnionType(resTypes);
    }

    mlir::Type getTypeByTypeQuery(TypeQueryNode typeQueryAST, const GenContext &genContext)
    {
        return getTypeByTypeName(typeQueryAST->exprName, genContext);
    }

    mlir::Type getConditionalType(ConditionalTypeNode conditionalTypeNode, const GenContext &genContext)
    {
        auto checkType = getType(conditionalTypeNode->checkType, genContext);
        auto extendsType = getType(conditionalTypeNode->extendsType, genContext);

        MLIRTypeHelper mth(builder.getContext());

        if (mth.extendsType(checkType, extendsType))
        {
            return getType(conditionalTypeNode->trueType, genContext);
        }

        mlir::Type resType;
        TypeSwitch<mlir::Type>(checkType)
            .template Case<mlir_ts::InterfaceType>([&](auto ifaceType) {
                auto interfaceInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue());
                assert(interfaceInfo);
                for (auto extend : interfaceInfo->extends)
                {
                    if (mth.extendsType(extend.second->interfaceType, extendsType))
                    {
                        resType = getType(conditionalTypeNode->trueType, genContext);
                        break;
                    }
                }
            })
            .template Case<mlir_ts::ClassType>([&](auto classType) {
                auto classInfo = getClassInfoByFullName(classType.getName().getValue());
                assert(classInfo);
                for (auto extend : classInfo->baseClasses)
                {
                    if (mth.extendsType(extend->classType, extendsType))
                    {
                        resType = getType(conditionalTypeNode->trueType, genContext);
                        break;
                    }
                }
            })
            .Default([&](auto type) { resType = getType(conditionalTypeNode->falseType, genContext); });

        return resType;
    }

    mlir::Type getKeyOf(TypeOperatorNode typeOperatorNode, const GenContext &genContext)
    {
        // this is "keyof"
        // TODO: finish it
        auto type = getType(typeOperatorNode->type, genContext);
        if (type.isa<mlir_ts::AnyType>())
        {
            return getUnionType(getStringType(), getNumberType());
        }

        if (type.isa<mlir_ts::UnknownType>())
        {
            return getNeverType();
        }

        if (type.isa<mlir_ts::ArrayType>())
        {
            return getNumberType();
        }

        if (auto objType = type.dyn_cast<mlir_ts::ObjectType>())
        {
            type = objType.getStorageType();
        }

        if (auto interfaceType = type.dyn_cast<mlir_ts::ClassType>())
        {
            auto classTypeInfo = getClassInfoByFullName(interfaceType.getName().getValue());
            type = classTypeInfo->classType.getStorageType();
        }

        if (auto tupleType = type.dyn_cast<mlir_ts::TupleType>())
        {
            SmallVector<mlir::Type> literalTypes;
            for (auto field : tupleType.getFields())
            {
                auto litType = mlir_ts::LiteralType::get(field.id, getAttributeType(field.id));
                literalTypes.push_back(litType);
            }

            if (literalTypes.size() == 1)
            {
                return literalTypes.front();
            }

            return getUnionType(literalTypes);
        }

        if (auto interfaceType = type.dyn_cast<mlir_ts::InterfaceType>())
        {
            auto interfaceTypeInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
            SmallVector<mlir::Type> literalTypes;
            for (auto field : interfaceTypeInfo->fields)
            {
                auto litType = mlir_ts::LiteralType::get(field.id, getAttributeType(field.id));
                literalTypes.push_back(litType);
            }

            if (literalTypes.size() == 1)
            {
                return literalTypes.front();
            }

            return getUnionType(literalTypes);
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! can't take 'keyof' from: " << type << "\n";);

        emitError(loc(typeOperatorNode), "can't take keyof: ") << type;

        return mlir::Type();
    }

    mlir::Type getTypeOperator(TypeOperatorNode typeOperatorNode, const GenContext &genContext)
    {
        if (typeOperatorNode->_operator == SyntaxKind::UniqueKeyword)
        {
            // TODO: finish it
            return getType(typeOperatorNode->type, genContext);
        }
        else if (typeOperatorNode->_operator == SyntaxKind::KeyOfKeyword)
        {
            return getKeyOf(typeOperatorNode, genContext);
        }

        llvm_unreachable("not implemented");
    }

    mlir::Type getAttributeType(mlir::Attribute attr)
    {
        if (attr.isa<mlir::StringAttr>())
        {
            return getStringType();
        }

        if (attr.isa<mlir::FloatAttr>())
        {
            return getNumberType();
        }

        llvm_unreachable("not implemented");
    }

    mlir::Type getIndexedAccessType(mlir::Type type, mlir::Type indexType, const GenContext &genContext)
    {
        if (auto unionType = indexType.dyn_cast<mlir_ts::UnionType>())
        {
            SmallVector<mlir::Type> resolvedTypes;
            for (auto itemType : unionType.getTypes())
            {
                auto resType = getIndexedAccessType(type, itemType, genContext);
                if (!resType)
                {
                    return mlir::Type();
                }

                resolvedTypes.push_back(resType);
            }

            return getUnionType(resolvedTypes);
        }

        if (auto objType = type.dyn_cast<mlir_ts::ObjectType>())
        {
            type = objType.getStorageType();
        }

        if (auto interfaceType = type.dyn_cast<mlir_ts::ClassType>())
        {
            auto classTypeInfo = getClassInfoByFullName(interfaceType.getName().getValue());
            type = classTypeInfo->classType.getStorageType();
        }

        if (auto tupleType = type.dyn_cast<mlir_ts::TupleType>())
        {
            SmallVector<mlir::Type> literalTypes;
            for (auto field : tupleType.getFields())
            {
                auto litType = mlir_ts::LiteralType::get(field.id, getAttributeType(field.id));

                LLVM_DEBUG(llvm::dbgs() << "\n!! field access type: " << litType << " <-> " << indexType << "\n";);

                if (litType == indexType)
                {
                    LLVM_DEBUG(llvm::dbgs() << "\n!! field access return type: " << field.type << "\n";);
                    return field.type;
                }
            }
        }

        if (auto interfaceType = type.dyn_cast<mlir_ts::InterfaceType>())
        {
            auto interfaceTypeInfo = getInterfaceInfoByFullName(interfaceType.getName().getValue());
            SmallVector<mlir::Type> literalTypes;
            for (auto field : interfaceTypeInfo->fields)
            {
                auto litType = mlir_ts::LiteralType::get(field.id, getAttributeType(field.id));
                if (litType == indexType)
                {
                    return field.type;
                }
            }
        }

        return mlir::Type();
    }

    mlir::Type getIndexedAccessType(IndexedAccessTypeNode indexedAccessTypeNode, const GenContext &genContext)
    {
        auto type = getType(indexedAccessTypeNode->objectType, genContext);
        auto indexType = getType(indexedAccessTypeNode->indexType, genContext);
        return getIndexedAccessType(type, indexType, genContext);
    }

    mlir::Type getTemplateLiteralType(TemplateLiteralTypeNode templateLiteralTypeNode, const GenContext &genContext)
    {
        auto location = loc(templateLiteralTypeNode);

        // first string
        auto text = convertWideToUTF8(templateLiteralTypeNode->head->rawText);

        SmallVector<mlir::Type> types;
        getTemplateLiteralSpan(types, text, templateLiteralTypeNode->templateSpans, 0, genContext);

        if (types.size() == 1)
        {
            return types.front();
        }

        return getUnionType(types);
    }

    void getTemplateLiteralSpan(SmallVector<mlir::Type> &types, std::string head, NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex,
                                const GenContext &genContext)
    {
        if (spanIndex >= spans.size())
        {
            auto newLiteralType = mlir_ts::LiteralType::get(builder.getStringAttr(head), getStringType());
            types.push_back(newLiteralType);
            return;
        }

        auto span = spans[spanIndex];
        auto type = getType(span->type, genContext);
        getTemplateLiteralTypeItem(types, type, head, spans, spanIndex, genContext);
    }

    void getTemplateLiteralTypeItem(SmallVector<mlir::Type> &types, mlir::Type type, std::string head,
                                    NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex, const GenContext &genContext)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! TemplateLiteralType, processing type: " << type << ", span: " << spanIndex << "\n";);

        if (auto unionType = type.dyn_cast<mlir_ts::UnionType>())
        {
            getTemplateLiteralUnionType(types, unionType, head, spans, spanIndex, genContext);
            return;
        }

        auto span = spans[spanIndex];

        std::stringstream ss;
        ss << head;

        auto typeText = type.cast<mlir_ts::LiteralType>().getValue().cast<mlir::StringAttr>().getValue();
        ss << typeText.str();

        auto spanText = convertWideToUTF8(span->literal->rawText);
        ss << spanText;

        getTemplateLiteralSpan(types, ss.str(), spans, spanIndex + 1, genContext);
    }

    void getTemplateLiteralUnionType(SmallVector<mlir::Type> &types, mlir::Type unionType, std::string head,
                                     NodeArray<TemplateLiteralTypeSpan> &spans, int spanIndex, const GenContext &genContext)
    {
        for (auto unionTypeItem : unionType.cast<mlir_ts::UnionType>().getTypes())
        {
            getTemplateLiteralTypeItem(types, unionTypeItem, head, spans, spanIndex, genContext);
        }
    }

    mlir::Type getMappedType(MappedTypeNode mappedTypeNode, const GenContext &genContext)
    {
        // PTR(Node) /**ReadonlyToken | PlusToken | MinusToken*/ readonlyToken;
        // PTR(TypeParameterDeclaration) typeParameter;
        // PTR(TypeNode) nameType;
        // PTR(Node) /**QuestionToken | PlusToken | MinusToken*/ questionToken;
        // PTR(TypeNode) type;

        auto typeParam = processTypeParameter(mappedTypeNode->typeParameter, genContext);
        auto hasNameType = !!mappedTypeNode->nameType;

        SmallVector<mlir_ts::FieldInfo> fields;
        for (auto typeParamItem : typeParam->getConstraint().cast<mlir_ts::UnionType>().getTypes())
        {
            const_cast<GenContext &>(genContext)
                .typeParamsWithArgs.insert({typeParam->getName(), std::make_pair(typeParam, typeParamItem)});

            auto type = getType(mappedTypeNode->type, genContext);

            mlir::Type nameType = typeParamItem;
            if (hasNameType)
            {
                nameType = getType(mappedTypeNode->nameType, genContext);
            }

            // remove type param
            const_cast<GenContext &>(genContext).typeParamsWithArgs.erase(typeParam->getName());

            LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type... type param: [" << typeParam->getName() << " constraint item: " << typeParamItem
                                    << ", name: " << nameType << "] type: " << type << "\n";);

            if (isNoneType(nameType) || nameType.isa<mlir_ts::NeverType>())
            {
                // filterting out
                LLVM_DEBUG(llvm::dbgs() << "\n!! mapped type... filtered.\n";);
                continue;
            }

            auto literalType = nameType.cast<mlir_ts::LiteralType>();
            fields.push_back({literalType.getValue(), type});
        }

        return getTupleType(fields);
    }

    mlir_ts::VoidType getVoidType()
    {
        return mlir_ts::VoidType::get(builder.getContext());
    }

    mlir_ts::ByteType getByteType()
    {
        return mlir_ts::ByteType::get(builder.getContext());
    }

    mlir_ts::BooleanType getBooleanType()
    {
        return mlir_ts::BooleanType::get(builder.getContext());
    }

    mlir_ts::NumberType getNumberType()
    {
        return mlir_ts::NumberType::get(builder.getContext());
    }

    mlir_ts::BigIntType getBigIntType()
    {
        return mlir_ts::BigIntType::get(builder.getContext());
    }

    mlir_ts::StringType getStringType()
    {
        return mlir_ts::StringType::get(builder.getContext());
    }

    mlir_ts::CharType getCharType()
    {
        return mlir_ts::CharType::get(builder.getContext());
    }

    bool isNoneType(mlir::Type type)
    {
        return !type || type == mlir::NoneType::get(builder.getContext());
    }

    bool isNotNoneType(mlir::Type type)
    {
        return !isNoneType(type);
    }

    mlir_ts::EnumType getEnumType()
    {
        return getEnumType(builder.getI32Type());
    }

    mlir_ts::EnumType getEnumType(mlir::Type elementType)
    {
        return mlir_ts::EnumType::get(elementType);
    }

    mlir_ts::ClassStorageType getClassStorageType(mlir::FlatSymbolRefAttr name)
    {
        return mlir_ts::ClassStorageType::get(builder.getContext(), name);
    }

    mlir_ts::ClassType getClassType(mlir::FlatSymbolRefAttr name, mlir::Type storageType)
    {
        return mlir_ts::ClassType::get(name, storageType);
    }

    mlir_ts::NamespaceType getNamespaceType(mlir::StringRef name)
    {
        auto nsNameAttr = mlir::FlatSymbolRefAttr::get(builder.getContext(), name);
        return mlir_ts::NamespaceType::get(nsNameAttr);
    }

    mlir_ts::InterfaceType getInterfaceType(StringRef fullName)
    {
        auto interfaceFullNameSymbol = mlir::FlatSymbolRefAttr::get(builder.getContext(), fullName);
        return getInterfaceType(interfaceFullNameSymbol);
    }

    mlir_ts::InterfaceType getInterfaceType(mlir::FlatSymbolRefAttr name)
    {
        return mlir_ts::InterfaceType::get(name);
    }

    mlir_ts::ConstArrayType getConstArrayType(ArrayTypeNode arrayTypeAST, unsigned size, const GenContext &genContext)
    {
        auto type = getType(arrayTypeAST->elementType, genContext);
        return getConstArrayType(type, size);
    }

    mlir_ts::ConstArrayType getConstArrayType(mlir::Type elementType, unsigned size)
    {
        assert(elementType);
        return mlir_ts::ConstArrayType::get(elementType, size);
    }

    mlir_ts::ArrayType getArrayType(ArrayTypeNode arrayTypeAST, const GenContext &genContext)
    {
        auto type = getType(arrayTypeAST->elementType, genContext);
        return getArrayType(type);
    }

    mlir_ts::ArrayType getArrayType(mlir::Type elementType)
    {
        return mlir_ts::ArrayType::get(elementType);
    }

    mlir_ts::ValueRefType getValueRefType(mlir::Type elementType)
    {
        return mlir_ts::ValueRefType::get(elementType);
    }

    mlir_ts::GenericType getGenericType()
    {
        return mlir_ts::GenericType::get(builder.getContext());
    }

    mlir_ts::NamedGenericType getNamedGenericType(StringRef name)
    {
        return mlir_ts::NamedGenericType::get(builder.getContext(), mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
    }

    mlir::Value getUndefined(mlir::Location location)
    {
        return builder.create<mlir_ts::UndefOp>(location, getOptionalType(getUndefPlaceHolderType()));
    }

    mlir::Value getInfinity(mlir::Location location)
    {
#ifdef NUMBER_F64
        double infVal;
        *(int64_t *)&infVal = 0x7FF0000000000000;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF64FloatAttr(infVal));
#else
        float infVal;
        *(int32_t *)&infVal = 0x7FF00000;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF32FloatAttr(infVal));
#endif
    }

    mlir::Value getNaN(mlir::Location location)
    {
#ifdef NUMBER_F64
        double nanVal;
        *(int64_t *)&nanVal = 0x7FF0000000000001;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF64FloatAttr(nanVal));
#else
        float infVal;
        *(int32_t *)&nanVal = 0x7FF00001;
        return builder.create<mlir_ts::ConstantOp>(location, getNumberType(), builder.getF32FloatAttr(nanVal));
#endif
    }

    void getTupleFieldInfo(TupleTypeNode tupleType, mlir::SmallVector<mlir_ts::FieldInfo> &types, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        mlir::Attribute attrVal;
        for (auto typeItem : tupleType->elements)
        {
            if (typeItem == SyntaxKind::NamedTupleMember)
            {
                auto namedTupleMember = typeItem.as<NamedTupleMember>();
                auto namePtr = MLIRHelper::getName(namedTupleMember->name, stringAllocator);

                auto type = getType(namedTupleMember->type, genContext);

                assert(type);
                types.push_back({mcl.TupleFieldName(namePtr), type});
            }
            else if (typeItem == SyntaxKind::LiteralType)
            {
                auto literalTypeNode = typeItem.as<LiteralTypeNode>();
                auto literalValue = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
                auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(literalValue.getDefiningOp());
                attrVal = constantOp.valueAttr();
                continue;
            }
            else
            {
                auto type = getType(typeItem, genContext);

                assert(type);
                types.push_back({attrVal, type});
            }

            attrVal = mlir::Attribute();
        }
    }

    void getTupleFieldInfo(TypeLiteralNode typeLiteral, mlir::SmallVector<mlir_ts::FieldInfo> &types, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        for (auto typeItem : typeLiteral->members)
        {
            if (typeItem == SyntaxKind::PropertySignature)
            {
                auto propertySignature = typeItem.as<PropertySignature>();
                auto namePtr = MLIRHelper::getName(propertySignature->name, stringAllocator);

                auto originalType = getType(propertySignature->type, genContext);
                auto type = mcl.getEffectiveFunctionTypeForTupleField(originalType);

                assert(type);
                types.push_back({mcl.TupleFieldName(namePtr), type});
            }
            else if (typeItem == SyntaxKind::MethodSignature)
            {
                auto methodSignature = typeItem.as<MethodSignature>();
                auto namePtr = MLIRHelper::getName(methodSignature->name, stringAllocator);

                auto type = getType(typeItem, genContext);

                assert(type);
                types.push_back({mcl.TupleFieldName(namePtr), type});
            }
            else
            {
                auto type = getType(typeItem, genContext);

                assert(type);
                types.push_back({mlir::Attribute(), type});
            }
        }
    }

    mlir_ts::ConstTupleType getConstTupleType(TupleTypeNode tupleType, const GenContext &genContext)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        getTupleFieldInfo(tupleType, types, genContext);
        return getConstTupleType(types);
    }

    mlir_ts::ConstTupleType getConstTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::ConstTupleType::get(builder.getContext(), fieldInfos);
    }

    mlir_ts::TupleType getTupleType(TupleTypeNode tupleType, const GenContext &genContext)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        getTupleFieldInfo(tupleType, types, genContext);
        return getTupleType(types);
    }

    mlir_ts::TupleType getTupleType(TypeLiteralNode typeLiteral, const GenContext &genContext)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        getTupleFieldInfo(typeLiteral, types, genContext);
        return getTupleType(types);
    }

    mlir_ts::TupleType getTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::TupleType::get(builder.getContext(), fieldInfos);
    }

    mlir_ts::ObjectType getObjectType(mlir::Type type)
    {
        return mlir_ts::ObjectType::get(type);
    }

    mlir_ts::BoundFunctionType getBoundFunctionType(mlir_ts::FunctionType funcType)
    {
        return mlir_ts::BoundFunctionType::get(builder.getContext(), funcType);
    }

    mlir_ts::BoundFunctionType getBoundFunctionType(ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> results, bool isVarArg = false)
    {
        return mlir_ts::BoundFunctionType::get(builder.getContext(), inputs, results, isVarArg);
    }

    mlir_ts::FunctionType getFunctionType(ArrayRef<mlir::Type> inputs, ArrayRef<mlir::Type> results, bool isVarArg = false)
    {
        return mlir_ts::FunctionType::get(builder.getContext(), inputs, results, isVarArg);
    }

    mlir_ts::FunctionType getSignature(SignatureDeclarationBase signature, const GenContext &genContext)
    {
        auto resultType = getType(signature->type, genContext);
        SmallVector<mlir::Type> argTypes;
        for (auto paramItem : signature->parameters)
        {
            auto type = getType(paramItem->type, genContext);
            if (paramItem->questionToken)
            {
                type = getOptionalType(type);
            }

            argTypes.push_back(type);
        }

        auto funcType = mlir_ts::FunctionType::get(builder.getContext(), argTypes, resultType);
        return funcType;
    }

    mlir_ts::HybridFunctionType getFunctionType(SignatureDeclarationBase signature, const GenContext &genContext)
    {
        auto funcType = mlir_ts::HybridFunctionType::get(builder.getContext(), getSignature(signature, genContext));
        return funcType;
    }

    mlir_ts::HybridFunctionType getConstructorType(SignatureDeclarationBase signature, const GenContext &genContext)
    {
        auto funcType = mlir_ts::HybridFunctionType::get(builder.getContext(), getSignature(signature, genContext));
        return funcType;
    }

    mlir_ts::HybridFunctionType getCallSignature(CallSignatureDeclaration signature, const GenContext &genContext)
    {
        auto funcType = mlir_ts::HybridFunctionType::get(builder.getContext(), getSignature(signature, genContext));
        return funcType;
    }

    mlir_ts::FunctionType getConstructSignature(ConstructSignatureDeclaration constructSignature, const GenContext &genContext)
    {
        return getSignature(constructSignature, genContext);
    }

    mlir_ts::FunctionType getMethodSignature(MethodSignature methodSignature, const GenContext &genContext)
    {
        return getSignature(methodSignature, genContext);
    }

    mlir_ts::FunctionType getIndexSignature(IndexSignatureDeclaration indexSignature, const GenContext &genContext)
    {
        return getSignature(indexSignature, genContext);
    }

    struct UnionTypeProcessContext
    {
        UnionTypeProcessContext() = default;

        bool isUndefined;
        bool isNullable;
        bool isNever;
        bool isAny;
        mlir::SmallPtrSet<mlir::Type, 2> types;
        mlir::SmallPtrSet<mlir::Type, 2> literalTypes;
    };

    mlir::LogicalResult processUnionTypeItem(mlir::Type type, UnionTypeProcessContext &unionContext, const GenContext &genContext)
    {
        if (type.isa<mlir_ts::UndefinedType>())
        {
            unionContext.isUndefined = true;
            return mlir::success();
        }

        if (type.isa<mlir_ts::NullType>())
        {
            unionContext.isNullable = true;
            return mlir::success();
        }

        if (type.isa<mlir_ts::AnyType>())
        {
            unionContext.isAny = true;
            return mlir::success();
        }

        if (type.isa<mlir_ts::NeverType>())
        {
            unionContext.isNever = true;
            return mlir::success();
        }

        if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
        {
            unionContext.literalTypes.insert(literalType);
            return mlir::success();
        }

        if (auto unionType = type.dyn_cast<mlir_ts::UnionType>())
        {
            if (mlir::succeeded(processUnionType(unionType, unionContext, genContext)))
            {
                return mlir::success();
            }
        }

        unionContext.types.insert(type);
        return mlir::success();
    }

    mlir::LogicalResult processUnionType(mlir_ts::UnionType unionType, UnionTypeProcessContext &unionContext, const GenContext &genContext)
    {
        for (auto type : unionType.getTypes())
        {
            processUnionTypeItem(type, unionContext, genContext);
        }

        return mlir::success();
    }

    mlir::Type getUnionType(UnionTypeNode unionTypeNode, const GenContext &genContext)
    {
        UnionTypeProcessContext unionContext = {};
        for (auto typeItem : unionTypeNode->types)
        {
            auto type = getType(typeItem, genContext);
            if (!type)
            {
                llvm_unreachable("wrong type");
            }

            processUnionTypeItem(type, unionContext, genContext);

            // default wide types
            if (unionContext.isAny)
            {
                return getAnyType();
            }

            if (unionContext.isNever)
            {
                return getNeverType();
            }
        }

        // merge types with literal types
        for (auto literalType : unionContext.literalTypes)
        {
            auto baseType = literalType.cast<mlir_ts::LiteralType>().getElementType();
            if (unionContext.types.count(baseType))
            {
                continue;
            }

            unionContext.types.insert(literalType);
        }

        MLIRTypeHelper mth(builder.getContext());
        auto isAllValueTypes = true;
        auto isAllLiteralTypes = true;

        mlir::SmallVector<mlir::Type> typesAll;
        for (auto type : unionContext.types)
        {
            typesAll.push_back(type);
            isAllValueTypes &= mth.isValueType(type);
            isAllLiteralTypes &= type.isa<mlir_ts::LiteralType>();
        }

        if ((isAllValueTypes || isAllLiteralTypes) && unionContext.isNullable)
        {
            // return null type back
            typesAll.push_back(getNullType());
        }

        mlir::Type retType = typesAll.size() == 1 ? typesAll.front() : getUnionType(typesAll);
        if (unionContext.isUndefined)
        {
            return getOptionalType(retType);
        }

        return retType;
    }

    mlir_ts::UnionType getUnionType(mlir::Type type1, mlir::Type type2)
    {
        mlir::SmallVector<mlir::Type> types;
        types.push_back(type1);
        types.push_back(type2);
        return mlir_ts::UnionType::get(builder.getContext(), types);
    }

    mlir::Type getUnionType(mlir::SmallVector<mlir::Type> &types)
    {
        if (types.size() == 0)
        {
            return getNeverType();
        }

        if (types.size() == 1)
        {
            return types.front();
        }

        return mlir_ts::UnionType::get(builder.getContext(), types);
    }

    mlir::Type getIntersectionType(IntersectionTypeNode intersectionTypeNode, const GenContext &genContext)
    {
        mlir_ts::InterfaceType baseInterfaceType;
        mlir_ts::TupleType baseTupleType;
        mlir::SmallVector<mlir::Type> types;
        mlir::Type firstNonFalseType;
        for (auto typeItem : intersectionTypeNode->types)
        {
            auto type = getType(typeItem, genContext);
            if (!type)
            {
                return getNeverType();
            }

            if (auto tupleType = type.dyn_cast<mlir_ts::TupleType>())
            {
                if (!baseTupleType)
                {
                    baseTupleType = tupleType;
                }
            }

            if (auto ifaceType = type.dyn_cast<mlir_ts::InterfaceType>())
            {
                if (!baseInterfaceType)
                {
                    baseInterfaceType = ifaceType;
                }
            }

            types.push_back(type);
        }

        if (types.size() == 0)
        {
            // this is never type
            return getNeverType();
        }

        if (types.size() == 1)
        {
            return types.front();
        }

        // find base type
        if (baseInterfaceType)
        {
            auto declareInterface = false;
            auto newInterfaceInfo = newInterfaceType(intersectionTypeNode, declareInterface, genContext);
            if (declareInterface)
            {
                // merge all interfaces;
                for (auto type : types)
                {
                    if (auto ifaceType = type.dyn_cast<mlir_ts::InterfaceType>())
                    {
                        auto srcInterfaceInfo = getInterfaceInfoByFullName(ifaceType.getName().getValue());
                        assert(srcInterfaceInfo);
                        newInterfaceInfo->extends.push_back({-1, srcInterfaceInfo});
                        continue;
                    }
                    else if (auto tupleType = type.dyn_cast<mlir_ts::TupleType>())
                    {
                        mergeInterfaces(newInterfaceInfo, tupleType);
                    }
                    else
                    {
                        // no intersection
                        return getNeverType();
                    }
                }
            }

            newInterfaceInfo->recalcOffsets();

            return newInterfaceInfo->interfaceType;
        }

        if (baseTupleType)
        {
            SmallVector<::mlir::typescript::FieldInfo> typesForNewTuple;
            for (auto type : types)
            {
                if (auto tupleType = type.dyn_cast<mlir_ts::TupleType>())
                {
                    for (auto field : tupleType.getFields())
                    {
                        typesForNewTuple.push_back(field);
                    }
                }
                else
                {
                    // no intersection
                    return getNeverType();
                }
            }

            return getTupleType(typesForNewTuple);
        }

        // calculate of intersaction between types and literal types
        mlir::Type resType;
        for (auto typeItem : types)
        {
            if (!resType)
            {
                resType = typeItem;
                continue;
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! &: " << resType << " & " << typeItem;);

            resType = AndType(resType, typeItem);

            LLVM_DEBUG(llvm::dbgs() << " = " << resType << "\n";);

            if (resType.isa<mlir_ts::NeverType>())
            {
                return getNeverType();
            }
        }

        if (resType)
        {
            return resType;
        }

        return getNeverType();
    }

    mlir::Type AndType(mlir::Type left, mlir::Type right)
    {
        // TODO: 00types_unknown1.ts contains examples of results with & | for types,  T & {} == T & {}, T | {} == T | {}, (they do not
        // change)
        if (left == right)
        {
            return left;
        }

        if (auto literalType = right.dyn_cast<mlir_ts::LiteralType>())
        {
            if (literalType.getElementType() == left)
            {
                if (left.isa<mlir_ts::LiteralType>())
                {
                    return getNeverType();
                }

                return literalType;
            }
        }

        if (auto leftUnionType = left.dyn_cast<mlir_ts::UnionType>())
        {
            return AndUnionType(leftUnionType, right);
        }

        if (auto unionType = right.dyn_cast<mlir_ts::UnionType>())
        {
            mlir::SmallPtrSet<mlir::Type, 2> newUniqueTypes;
            for (auto unionTypeItem : unionType.getTypes())
            {
                auto resType = AndType(left, unionTypeItem);
                newUniqueTypes.insert(resType);
            }

            SmallVector<mlir::Type> newTypes;
            for (auto uniqType : newUniqueTypes)
            {
                newTypes.push_back(uniqType);
            }

            return getUnionType(newTypes);
        }

        if (left.isa<mlir_ts::NullType>())
        {
            MLIRTypeHelper mth(builder.getContext());
            if (mth.isValueType(right))
            {
                return getNeverType();
            }

            return left;
        }

        if (right.isa<mlir_ts::NullType>())
        {
            MLIRTypeHelper mth(builder.getContext());
            if (mth.isValueType(left))
            {
                return getNeverType();
            }

            return right;
        }

        if (left.isa<mlir_ts::NullType>())
        {
            MLIRTypeHelper mth(builder.getContext());
            if (mth.isValueType(right))
            {
                return getNeverType();
            }

            return left;
        }

        if (left.isa<mlir_ts::AnyType>() || left.isa<mlir_ts::UnknownType>())
        {
            return right;
        }

        if (right.isa<mlir_ts::AnyType>() || right.isa<mlir_ts::UnknownType>())
        {
            return left;
        }

        // TODO: should I add, interface, tuple types here?
        // PS: string & { __b: number } creating type "string & { __b: number }".

        return mlir::Type();
    }

    mlir::Type AndUnionType(mlir_ts::UnionType leftUnion, mlir::Type right)
    {
        mlir::SmallPtrSet<mlir::Type, 2> newUniqueTypes;
        for (auto unionTypeItem : leftUnion.getTypes())
        {
            auto resType = AndType(unionTypeItem, right);
            newUniqueTypes.insert(resType);
        }

        SmallVector<mlir::Type> newTypes;
        for (auto uniqType : newUniqueTypes)
        {
            newTypes.push_back(uniqType);
        }

        return getUnionType(newTypes);
    }

    InterfaceInfo::TypePtr newInterfaceType(IntersectionTypeNode intersectionTypeNode, bool &declareInterface, const GenContext &genContext)
    {
        auto newName = MLIRHelper::getAnonymousName(loc_check(intersectionTypeNode), "ifce");

        // clone into new interface
        auto interfaceInfo = mlirGenInterfaceInfo(newName, declareInterface, genContext);

        return interfaceInfo;
    }

    mlir::LogicalResult mergeInterfaces(InterfaceInfo::TypePtr dest, mlir_ts::TupleType src)
    {
        // TODO: use it to merge with TupleType
        for (auto &item : src.getFields())
        {
            dest->fields.push_back({item.id, item.type, false, dest->getNextVTableMemberIndex()});
        }

        return mlir::success();
    }

    mlir::Type getParenthesizedType(ParenthesizedTypeNode parenthesizedTypeNode, const GenContext &genContext)
    {
        return getType(parenthesizedTypeNode->type, genContext);
    }

    mlir::Type getLiteralType(LiteralTypeNode literalTypeNode)
    {
        GenContext genContext{};
        genContext.dummyRun = true;
        genContext.allowPartialResolve = true;
        auto value = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
        auto type = value.getType();

        auto constantOp = value.getDefiningOp<mlir_ts::ConstantOp>();
        if (constantOp)
        {
            auto valueAttr = value.getDefiningOp<mlir_ts::ConstantOp>().valueAttr();
            auto literalType = mlir_ts::LiteralType::get(valueAttr, type);
            return literalType;
        }

        auto nullOp = value.getDefiningOp<mlir_ts::NullOp>();
        if (nullOp)
        {
            return getNullType();
        }

        LLVM_DEBUG(llvm::dbgs() << "\n!! value of literal: " << value << "\n";);

        llvm_unreachable("not implemented");
    }

    mlir_ts::OptionalType getOptionalType(mlir::Type type)
    {
        return mlir_ts::OptionalType::get(type);
    }

    mlir_ts::UndefPlaceHolderType getUndefPlaceHolderType()
    {
        return mlir_ts::UndefPlaceHolderType::get(builder.getContext());
    }

    mlir_ts::AnyType getAnyType()
    {
        return mlir_ts::AnyType::get(builder.getContext());
    }

    mlir_ts::UnknownType getUnknownType()
    {
        return mlir_ts::UnknownType::get(builder.getContext());
    }

    mlir_ts::NeverType getNeverType()
    {
        return mlir_ts::NeverType::get(builder.getContext());
    }

    mlir_ts::SymbolType getSymbolType()
    {
        return mlir_ts::SymbolType::get(builder.getContext());
    }

    mlir_ts::UndefinedType getUndefinedType()
    {
        return mlir_ts::UndefinedType::get(builder.getContext());
    }

    mlir_ts::NullType getNullType()
    {
        return mlir_ts::NullType::get(builder.getContext());
    }

    mlir_ts::OpaqueType getOpaqueType()
    {
        return mlir_ts::OpaqueType::get(builder.getContext());
    }

    mlir::LogicalResult declare(VariableDeclarationDOM::TypePtr var, mlir::Value value, const GenContext &genContext,
                                bool redefineVar = false)
    {
        const auto &name = var->getName();
        /*
        if (symbolTable.count(name))
        {
            return mlir::failure();
        }
        */

        if (!genContext.insertIntoParentScope)
        {
            symbolTable.insert(name, {value, var});
        }
        else
        {
            symbolTable.insertIntoScope(symbolTable.getCurScope()->getParentScope(), name, {value, var});
        }

        return mlir::success();
    }

    auto getNamespace() -> StringRef
    {
        if (currentNamespace->fullName.empty())
        {
            return "";
        }

        return currentNamespace->fullName;
    }

    auto getFullNamespaceName(StringRef name) -> StringRef
    {
        if (currentNamespace->fullName.empty())
        {
            return name;
        }

        std::string res;
        res += currentNamespace->fullName;
        res += ".";
        res += name;

        auto namePtr = StringRef(res).copy(stringAllocator);
        return namePtr;
    }

    auto concat(StringRef fullNamespace, StringRef name) -> StringRef
    {
        std::string res;
        res += fullNamespace;
        res += ".";
        res += name;

        auto namePtr = StringRef(res).copy(stringAllocator);
        return namePtr;
    }

    auto concat(StringRef fullNamespace, StringRef className, StringRef name) -> StringRef
    {
        std::string res;
        res += fullNamespace;
        res += ".";
        res += className;
        res += ".";
        res += name;

        auto namePtr = StringRef(res).copy(stringAllocator);
        return namePtr;
    }

    auto getNamespaceByFullName(StringRef fullName) -> NamespaceInfo::TypePtr
    {
        return fullNamespacesMap.lookup(fullName);
    }

    auto getNamespaceMap() -> llvm::StringMap<NamespaceInfo::TypePtr> &
    {
        return currentNamespace->namespacesMap;
    }

    auto getFunctionMap() -> llvm::StringMap<mlir_ts::FuncOp> &
    {
        return currentNamespace->functionMap;
    }

    auto getGenericFunctionMap() -> llvm::StringMap<GenericFunctionInfo::TypePtr> &
    {
        return currentNamespace->genericFunctionMap;
    }

    auto getGlobalsMap() -> llvm::StringMap<VariableDeclarationDOM::TypePtr> &
    {
        return currentNamespace->globalsMap;
    }

    auto getCaptureVarsMap() -> llvm::StringMap<llvm::StringMap<ts::VariableDeclarationDOM::TypePtr>> &
    {
        return currentNamespace->captureVarsMap;
    }

    auto getLocalVarsInThisContextMap() -> llvm::StringMap<llvm::SmallVector<mlir::typescript::FieldInfo>> &
    {
        return currentNamespace->localVarsInThisContextMap;
    }

    auto getClassesMap() -> llvm::StringMap<ClassInfo::TypePtr> &
    {
        return currentNamespace->classesMap;
    }

    auto getGenericClassesMap() -> llvm::StringMap<GenericClassInfo::TypePtr> &
    {
        return currentNamespace->genericClassesMap;
    }

    auto getInterfacesMap() -> llvm::StringMap<InterfaceInfo::TypePtr> &
    {
        return currentNamespace->interfacesMap;
    }

    auto getGenericInterfacesMap() -> llvm::StringMap<GenericInterfaceInfo::TypePtr> &
    {
        return currentNamespace->genericInterfacesMap;
    }

    auto getEnumsMap() -> llvm::StringMap<std::pair<mlir::Type, mlir::DictionaryAttr>> &
    {
        return currentNamespace->enumsMap;
    }

    auto getTypeAliasMap() -> llvm::StringMap<mlir::Type> &
    {
        return currentNamespace->typeAliasMap;
    }

    auto getGenericTypeAliasMap() -> llvm::StringMap<std::pair<llvm::SmallVector<TypeParameterDOM::TypePtr>, TypeNode>> &
    {
        return currentNamespace->genericTypeAliasMap;
    }

    auto getImportEqualsMap() -> llvm::StringMap<mlir::StringRef> &
    {
        return currentNamespace->importEqualsMap;
    }

    auto getClassInfoByFullName(StringRef fullName) -> ClassInfo::TypePtr
    {
        return fullNameClassesMap.lookup(fullName);
    }

    auto hasGenericClassInfoByFullName(StringRef fullName) -> bool
    {
        return fullNameGenericClassesMap.count(fullName) > 0;
    }

    auto getGenericClassInfoByFullName(StringRef fullName) -> GenericClassInfo::TypePtr
    {
        return fullNameGenericClassesMap.lookup(fullName);
    }

    auto getInterfaceInfoByFullName(StringRef fullName) -> InterfaceInfo::TypePtr
    {
        return fullNameInterfacesMap.lookup(fullName);
    }

    auto hasGenericInterfaceInfoByFullName(StringRef fullName) -> bool
    {
        return fullNameGenericInterfacesMap.count(fullName) > 0;
    }

    auto getGenericInterfaceInfoByFullName(StringRef fullName) -> GenericInterfaceInfo::TypePtr
    {
        return fullNameGenericInterfacesMap.lookup(fullName);
    }

  protected:
    mlir::StringAttr getStringAttr(std::string text)
    {
        return builder.getStringAttr(text);
    }

    /// Helper conversion for a TypeScript AST location to an MLIR location.
    mlir::Location loc(TextRange loc)
    {
        if (!loc)
        {
            return mlir::UnknownLoc::get(builder.getContext());
        }

        // return builder.getFileLineColLoc(builder.getIdentifier(fileName), loc->pos, loc->_end);
        auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, loc->pos.textPos != -1 ? loc->pos.textPos : loc->pos.pos);
        return mlir::FileLineColLoc::get(builder.getContext(), builder.getIdentifier(fileName), posLineChar.line + 1,
                                         posLineChar.character + 1);
    }

    mlir::Location loc_check(TextRange loc_)
    {
        assert(loc_->pos != loc_->_end);
        return loc(loc_);
    }

    bool hasErrorMessages;

    /// The builder is a helper class to create IR inside a function. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder;

    CompileOptions compileOptions;

    /// A "module" matches a TypeScript source file: containing a list of functions.
    mlir::ModuleOp theModule;

    mlir::StringRef fileName;

    /// An allocator used for alias names.
    llvm::BumpPtrAllocator stringAllocator;

    llvm::ScopedHashTable<StringRef, VariablePairT> symbolTable;

    NamespaceInfo::TypePtr rootNamespace;

    NamespaceInfo::TypePtr currentNamespace;

    llvm::ScopedHashTable<StringRef, NamespaceInfo::TypePtr> fullNamespacesMap;

    llvm::ScopedHashTable<StringRef, ClassInfo::TypePtr> fullNameClassesMap;

    llvm::ScopedHashTable<StringRef, GenericClassInfo::TypePtr> fullNameGenericClassesMap;

    llvm::ScopedHashTable<StringRef, InterfaceInfo::TypePtr> fullNameInterfacesMap;

    llvm::ScopedHashTable<StringRef, GenericInterfaceInfo::TypePtr> fullNameGenericInterfacesMap;

    llvm::ScopedHashTable<StringRef, VariableDeclarationDOM::TypePtr> fullNameGlobalsMap;

    // helper to get line number
    Parser parser;
    ts::SourceFile sourceFile;

    mlir::OpBuilder::InsertPoint functionBeginPoint;

    std::string label;
};
} // namespace

namespace typescript
{
::std::string dumpFromSource(const llvm::StringRef &fileName, const llvm::StringRef &source)
{
    auto showLineCharPos = false;

    Parser parser;
    auto sourceFile =
        parser.parseSourceFile(stows(static_cast<std::string>(fileName)), stows(static_cast<std::string>(source)), ScriptTarget::Latest);

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

            s << S("Node: ") << parser.syntaxKindString(child).c_str() << S(" @ [ ") << child->pos << S("(") << posLineChar.line + 1
              << S(":") << posLineChar.character + 1 << S(") - ") << child->_end << S("(") << endLineChar.line + 1 << S(":")
              << endLineChar.character << S(") ]") << std::endl;
        }
        else
        {
            s << S("Node: ") << parser.syntaxKindString(child).c_str() << S(" @ [ ") << child->pos << S(" - ") << child->_end << S(" ]")
              << std::endl;
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

mlir::OwningModuleRef mlirGenFromSource(const mlir::MLIRContext &context, const llvm::StringRef &fileName, const llvm::StringRef &source,
                                        CompileOptions compileOptions)
{
    Parser parser;
    auto sourceFile =
        parser.parseSourceFile(stows(static_cast<std::string>(fileName)), stows(static_cast<std::string>(source)), ScriptTarget::Latest);
    return MLIRGenImpl(context, fileName, compileOptions).mlirGenSourceFile(sourceFile);
}

} // namespace typescript
