#define DEBUG_TYPE "mlir"

#include "TypeScript/MLIRGen.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "TypeScript/MLIRGenLogic.h"
#include "TypeScript/VisitorAST.h"

#include "TypeScript/DOM.h"
#include "TypeScript/Defines.h"

// parser includes
#include "file_helper.h"
#include "node_factory.h"
#include "parser.h"
#include "utilities.h"

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

#define VALIDATE_EXPR(value, expression)                                                                                                   \
    if (!value)                                                                                                                            \
    {                                                                                                                                      \
        if (!genContext.allowPartialResolve)                                                                                               \
        {                                                                                                                                  \
            emitError(loc(expression), "expression has no result");                                                                        \
        }                                                                                                                                  \
                                                                                                                                           \
        return mlir::Value();                                                                                                              \
    }                                                                                                                                      \
                                                                                                                                           \
    if (auto unresolved = dyn_cast_or_null<mlir_ts::SymbolRefOp>(value.getDefiningOp()))                                                   \
    {                                                                                                                                      \
        if (!genContext.allowPartialResolve)                                                                                               \
        {                                                                                                                                  \
            emitError(loc(expression), "can't find variable: ") << unresolved.identifier();                                                \
        }                                                                                                                                  \
                                                                                                                                           \
        return mlir::Value();                                                                                                              \
    }

namespace
{

enum class VariableClass
{
    Const,
    Let,
    Var
};

struct PassResult
{
    mlir::Type functionReturnType;
    llvm::StringMap<VariablePairT> outerVariables;
};

struct GenContext
{
    GenContext() = default;

    // TODO: you are using "theModule.getBody()->clear();", do you need this hack anymore?
    void clean()
    {
        if (cleanUps)
        {
            for (auto op : *cleanUps)
            {
                op->erase();
            }

            delete cleanUps;
            cleanUps = nullptr;
        }

        if (passResult)
        {
            delete passResult;
            passResult = nullptr;
        }
    }

    bool allowPartialResolve;
    bool dummyRun;
    bool allowConstEval;
    mlir_ts::FuncOp funcOp;
    PassResult *passResult;
    mlir::SmallVector<mlir::Block *> *cleanUps;
};

struct ClassStaticFieldInfo
{
    Attribute id;
    mlir::Type type;
};

struct ClassInfo
{
  public:
    using TypePtr = std::shared_ptr<ClassInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    mlir::Type storageType;

    llvm::StringMap<ClassStaticFieldInfo> staticFields;
};

struct NamespaceInfo
{
  public:
    using TypePtr = std::shared_ptr<NamespaceInfo>;

    mlir::StringRef name;

    mlir::StringRef fullName;

    llvm::StringMap<mlir_ts::FuncOp> functionMap;

    llvm::StringMap<VariableDeclarationDOM::TypePtr> globalsMap;

    llvm::StringMap<llvm::StringMap<VariablePairT>> captureVarsMap;

    llvm::StringMap<mlir::Type> typeAliasMap;

    llvm::StringMap<mlir::StringRef> importEqualsMap;

    llvm::StringMap<std::pair<mlir::Type, mlir::DictionaryAttr>> enumsMap;

    llvm::StringMap<ClassInfo::TypePtr> classesMap;

    llvm::StringMap<NamespaceInfo::TypePtr> namespacesMap;
};

/// Implementation of a simple MLIR emission from the TypeScript AST.
///
/// This will emit operations that are specific to the TypeScript language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl
{
  public:
    MLIRGenImpl(const mlir::MLIRContext &context) : builder(&const_cast<mlir::MLIRContext &>(context))
    {
        fileName = "<unknown>";
        rootNamespace = currentNamespace = std::make_shared<NamespaceInfo>();
    }

    MLIRGenImpl(const mlir::MLIRContext &context, const llvm::StringRef &fileNameParam) : builder(&const_cast<mlir::MLIRContext &>(context))
    {
        fileName = fileNameParam;
        rootNamespace = currentNamespace = std::make_shared<NamespaceInfo>();
    }

    mlir::ModuleOp mlirGenSourceFile(SourceFile module)
    {
        sourceFile = module;

        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        theModule = mlir::ModuleOp::create(loc(module), fileName);
        builder.setInsertionPointToStart(theModule.getBody());

        SymbolTableScopeT varScope(symbolTable);

        // Process of discovery here
        GenContext genContextPartial = {0};
        genContextPartial.allowPartialResolve = true;
        genContextPartial.dummyRun = true;
        genContextPartial.cleanUps = new mlir::SmallVector<mlir::Block *>();
        auto notAllResolved = false;
        do
        {
            GenContext genContext = {0};
            for (auto &statement : module->statements)
            {
                if (statement->processed)
                {
                    continue;
                }

                if (failed(mlirGen(statement, genContextPartial)))
                {
                    notAllResolved = true;
                }
                else
                {
                    statement->processed = true;
                }
            }
        } while (notAllResolved);

        genContextPartial.clean();

        // clean up
        theModule.getBody()->clear();

        // Process generating here
        GenContext genContext = {0};
        for (auto &statement : module->statements)
        {
            if (failed(mlirGen(statement, genContext)))
            {
                return nullptr;
            }
        }

        // Verify the module after we have finished constructing it, this will check
        // the structural properties of the IR and invoke any specific verifiers we
        // have on the TypeScript operations.
        if (failed(mlir::verify(theModule)))
        {
            // TODO: uncomment it
            theModule.emitError("module verification error");
            // return nullptr;
        }

        return theModule;
    }

    mlir::LogicalResult mlirGenNamespace(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext)
    {
        auto location = loc(moduleDeclarationAST);

        auto namespaceName = MLIRHelper::getName(moduleDeclarationAST->name);
        auto namePtr = StringRef(namespaceName).copy(stringAllocator);

        auto savedNamespace = currentNamespace;

        auto fullNamePtr = getFullNamespaceName(namePtr);
        auto &namespacesMap = getNamespaceMap();
        auto it = namespacesMap.find(namePtr);
        if (it == namespacesMap.end())
        {
            auto newNamespacePtr = std::make_shared<NamespaceInfo>();
            newNamespacePtr->name = namePtr;
            newNamespacePtr->fullName = fullNamePtr;
            namespacesMap.insert({namePtr, newNamespacePtr});
            fullNamespacesMap.insert({fullNamePtr, newNamespacePtr});
            currentNamespace = newNamespacePtr;
        }
        else
        {
            currentNamespace = it->getValue();
        }

        GenContext moduleGenContext = {0};
        auto result = mlirGenBody(moduleDeclarationAST->body, genContext);

        currentNamespace = savedNamespace;

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ModuleDeclaration moduleDeclarationAST, const GenContext &genContext)
    {
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

        GenContext moduleGenContext = {0};
        auto result = mlirGenBody(moduleDeclarationAST->body, genContext);

        // restore
        theModule = parentModule;

        builder.setInsertionPointAfter(moduleOp);

        return result;
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
            return mlirGenReturnValue(loc(body), result, genContext);
        }

        llvm_unreachable("unknown body type");
    }

    mlir::LogicalResult mlirGen(ModuleBlock moduleBlockAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        for (auto &statement : moduleBlockAST->statements)
        {
            if (genContext.dummyRun && statement->processed)
            {
                continue;
            }

            if (failed(mlirGen(statement, genContext)))
            {
                return mlir::failure();
            }

            if (genContext.dummyRun)
            {
                statement->processed = true;
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(Block blockAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        for (auto &statement : blockAST->statements)
        {
            if (genContext.dummyRun && statement->processed)
            {
                continue;
            }

            if (failed(mlirGen(statement, genContext)))
            {
                return mlir::failure();
            }

            if (genContext.dummyRun)
            {
                statement->processed = true;
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
        else if (kind == SyntaxKind::ImportEqualsDeclaration)
        {
            // declaration
            return mlirGen(statementAST.as<ImportEqualsDeclaration>(), genContext);
        }
        else if (kind == SyntaxKind::ModuleDeclaration)
        {
            return mlirGen(statementAST.as<ModuleDeclaration>(), genContext);
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
        else if (kind == SyntaxKind::VoidExpression)
        {
            return mlirGen(expressionAST.as<VoidExpression>(), genContext);
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

    bool registerVariable(mlir::Location location, StringRef name, VariableClass varClass,
                          std::function<std::pair<mlir::Type, mlir::Value>()> func, const GenContext &genContext)
    {
        auto isGlobalScope = !genContext.funcOp; /*symbolTable.getCurScope()->getParentScope() == nullptr*/
        auto isGlobal = isGlobalScope || varClass == VariableClass::Var;
        auto isConst = varClass == VariableClass::Const;

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
            }
            else
            {
                assert(type);

                MLIRTypeHelper mth(builder.getContext());

                auto copyRequired = false;
                auto actualType = mth.convertConstTypeToType(type, copyRequired);
                if (init && actualType != type)
                {
                    auto castValue = builder.create<mlir_ts::CastOp>(location, actualType, init);
                    init = castValue;
                }

                variableOp = builder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(actualType), init);
            }
        }
        else
        {
            mlir_ts::GlobalOp globalOp;
            // get constant
            {
                mlir::OpBuilder::InsertionGuard insertGuard(builder);
                builder.setInsertionPointToStart(theModule.getBody());

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

                // save value
                auto address = builder.create<mlir_ts::AddressOfOp>(location, mlir_ts::RefType::get(type), name);
                builder.create<mlir_ts::StoreOp>(location, init, address);
            }
        }

        // registering variable
        auto varDecl = std::make_shared<VariableDeclarationDOM>(effectiveName, varType, location);
        if (!isConst)
        {
            varDecl->setReadWriteAccess();
        }

        varDecl->setFuncOp(genContext.funcOp);

        if (!isGlobal)
        {
            declare(varDecl, variableOp);
        }
        else
        {
            getGlobalsMap().insert({name, varDecl});
        }

        return true;
    }

    template <typename ItemTy>
    bool processDeclaration(ItemTy item, VariableClass varClass, std::function<std::pair<mlir::Type, mlir::Value>()> func,
                            const GenContext &genContext)
    {
        auto location = loc(item);

        if (item->name == SyntaxKind::ArrayBindingPattern)
        {
            auto res = func();
            auto type = std::get<0>(res);
            auto init = std::get<1>(res);

            auto arrayBindingPattern = item->name.template as<ArrayBindingPattern>();
            auto index = 0;
            for (auto arrayBindingElement : arrayBindingPattern->elements)
            {
                MLIRPropertyAccessCodeLogic cl(builder, location, init, builder.getI32IntegerAttr(index++));
                mlir::Value subInit;
                TypeSwitch<mlir::Type>(type)
                    .template Case<mlir_ts::ConstTupleType>([&](auto constTupleType) { subInit = cl.Tuple(constTupleType, true); })
                    .template Case<mlir_ts::TupleType>([&](auto tupleType) { subInit = cl.Tuple(tupleType, true); })
                    .Default([&](auto type) { llvm_unreachable("not implemented"); });

                if (!processDeclaration(
                        arrayBindingElement.template as<BindingElement>(), varClass,
                        [&]() { return std::make_pair(subInit.getType(), subInit); }, genContext))
                {
                    return false;
                }
            }
        }
        else
        {
            // name
            auto name = MLIRHelper::getName(item->name);

            // register
            return registerVariable(location, name, varClass, func, genContext);
        }

        return true;
    }

    mlir::LogicalResult mlirGen(VariableDeclarationList variableDeclarationListAST, const GenContext &genContext)
    {
        auto isLet = (variableDeclarationListAST->flags & NodeFlags::Let) == NodeFlags::Let;
        auto isConst = (variableDeclarationListAST->flags & NodeFlags::Const) == NodeFlags::Const;
        auto varClass = isLet ? VariableClass::Let : isConst ? VariableClass::Const : VariableClass::Var;

        for (auto &item : variableDeclarationListAST->declarations)
        {
            auto initFunc = [&]() {
                // type
                mlir::Type type;
                if (item->type)
                {
                    type = getType(item->type);
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
                            auto castValue = builder.create<mlir_ts::CastOp>(loc(initializer), type, init);
                            init = castValue;
                        }
                    }
                }

                return std::make_pair(type, init);
            };

            auto valClassItem = varClass;
            if ((item->transformFlags & TransformFlags::ForceConst) == TransformFlags::ForceConst)
            {
                valClassItem = VariableClass::Const;
            }

            if (!processDeclaration(item, valClassItem, initFunc, genContext))
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

    std::vector<std::shared_ptr<FunctionParamDOM>> mlirGenParameters(SignatureDeclarationBase parametersContextAST,
                                                                     const GenContext &genContext)
    {
        std::vector<std::shared_ptr<FunctionParamDOM>> params;
        if (!parametersContextAST)
        {
            return params;
        }

        auto formalParams = parametersContextAST->parameters;
        for (auto arg : formalParams)
        {
            auto name = MLIRHelper::getName(arg->name);
            mlir::Type type;
            auto isOptional = !!arg->questionToken;
            auto typeParameter = arg->type;
            if (typeParameter)
            {
                type = getType(typeParameter);
                if (!type)
                {
                    if (!genContext.allowPartialResolve)
                    {
                        emitError(loc(typeParameter)) << "can't resolve type for parameter '" << name << "'";
                    }

                    return params;
                }
            }

            // process init value
            auto initializer = arg->initializer;
            if (initializer)
            {
                // we need to add temporary block
                auto tempFuncType = builder.getFunctionType(llvm::None, llvm::None);
                auto tempFuncOp = mlir::FuncOp::create(loc(initializer), name, tempFuncType);
                auto &entryBlock = *tempFuncOp.addEntryBlock();

                auto insertPoint = builder.saveInsertionPoint();
                builder.setInsertionPointToStart(&entryBlock);

                auto initValue = mlirGen(initializer, genContext);
                if (initValue)
                {
                    // TODO: set type if not provided
                    isOptional = true;
                    if (!type)
                    {
                        auto baseType = initValue.getType();
                        // type = OptionalType::get(baseType);
                        type = baseType;
                    }
                }

                // remove temp block
                builder.restoreInsertionPoint(insertPoint);
                entryBlock.erase();
            }

            if (!typeParameter && !initializer)
            {
                auto funcName = MLIRHelper::getName(parametersContextAST->name);
                emitError(loc(arg)) << "type of parameter '" << name
                                    << "' is not provided, parameter must have type or initializer, function: " << funcName;
            }

            params.push_back(std::make_shared<FunctionParamDOM>(name, type, loc(arg), isOptional, initializer));
        }

        return params;
    }

    std::tuple<mlir_ts::FuncOp, FunctionPrototypeDOM::TypePtr, bool> mlirGenFunctionPrototype(
        FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        auto location = loc(functionLikeDeclarationBaseAST);

        std::vector<FunctionParamDOM::TypePtr> params = mlirGenParameters(functionLikeDeclarationBaseAST, genContext);
        SmallVector<mlir::Type> argTypes;
        auto argNumber = 0;

        for (const auto &param : params)
        {
            auto paramType = param->getType();
            if (!paramType)
            {
                return std::make_tuple(mlir_ts::FuncOp(), FunctionPrototypeDOM::TypePtr(nullptr), false);
            }

            if (param->getIsOptional())
            {
                argTypes.push_back(getOptionalType(paramType));
            }
            else
            {
                argTypes.push_back(paramType);
            }

            argNumber++;
        }

        auto fullName = MLIRHelper::getName(functionLikeDeclarationBaseAST->name);
        auto name = fullName;
        if (fullName.empty())
        {
            // auto calculate name
            std::stringstream ssName;
            ssName << "__uf" << hash_value(location);
            name = fullName = ssName.str();
        }
        else
        {
            fullName = getFullNamespaceName(name);
        }

        auto funcProto = std::make_shared<FunctionPrototypeDOM>(fullName, params);

        mlir::FunctionType funcType;

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

        // discover type & args
        if (!funcType)
        {
            if (auto typeParameter = functionLikeDeclarationBaseAST->type)
            {
                auto returnType = getType(typeParameter);
                funcProto->setReturnType(returnType);
            }

            if (mlir::succeeded(
                    discoverFunctionReturnTypeAndCapturedVars(functionLikeDeclarationBaseAST, fullName, argTypes, funcProto, genContext)) &&
                funcProto->getReturnType())
            {
                funcType = builder.getFunctionType(argTypes, funcProto->getReturnType());
            }
            else
            {
                // no return type
                funcType = builder.getFunctionType(argTypes, llvm::None);
            }
        }

        auto it = getCaptureVarsMap().find(funcProto->getName());
        auto hasCapturedVars = funcProto->getHasCapturedVars() || (it != getCaptureVarsMap().end());

        mlir_ts::FuncOp funcOp;
        if (hasCapturedVars)
        {
            SmallVector<mlir::NamedAttribute> attrs;
            SmallVector<mlir::DictionaryAttr> argAttrs;

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

            LLVM_DEBUG(llvm::dbgs() << "\n === FuncOp with attrs === \n");
            LLVM_DEBUG(funcOp.dump());
        }
        else
        {
            funcOp = mlir_ts::FuncOp::create(location, fullName, funcType);
        }

        return std::make_tuple(funcOp, std::move(funcProto), true);
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

        LLVM_DEBUG(llvm::dbgs() << "discover func ret type for : " << name << "\n";);

        mlir::OpBuilder::InsertionGuard guard(builder);

        auto partialDeclFuncType = builder.getFunctionType(argTypes, llvm::None);
        auto dummyFuncOp = mlir_ts::FuncOp::create(loc(functionLikeDeclarationBaseAST), name, partialDeclFuncType);

        {
            // simulate scope
            SymbolTableScopeT varScope(symbolTable);

            GenContext genContextWithPassResult = {0};
            genContextWithPassResult.funcOp = dummyFuncOp;
            genContextWithPassResult.allowPartialResolve = true;
            genContextWithPassResult.dummyRun = true;
            genContextWithPassResult.cleanUps = new SmallVector<mlir::Block *>();
            genContextWithPassResult.passResult = new PassResult();
            if (succeeded(mlirGenFunctionBody(functionLikeDeclarationBaseAST, dummyFuncOp, funcProto, genContextWithPassResult)))
            {
                funcProto->setDiscovered(true);
                auto discoveredType = genContextWithPassResult.passResult->functionReturnType;
                if (discoveredType && discoveredType != funcProto->getReturnType())
                {
                    // TODO: do we need to convert it here? maybe send it as const object?
                    MLIRTypeHelper mth(builder.getContext());
                    bool copyRequired;
                    funcProto->setReturnType(mth.convertConstTypeToType(discoveredType, copyRequired));
                    LLVM_DEBUG(llvm::dbgs() << "ret type for " << name << " : " << funcProto->getReturnType() << "\n";);
                }

                // if we have captured parameters, add first param to send lambda's type(class)
                if (genContextWithPassResult.passResult->outerVariables.size() > 0)
                {
                    MLIRCodeLogic mcl(builder);
                    argTypes.insert(argTypes.begin(), mcl.CaptureType(genContextWithPassResult.passResult->outerVariables));
                    getCaptureVarsMap().insert({name, genContextWithPassResult.passResult->outerVariables});

                    funcProto->setHasCapturedVars(true);
                }
            }

            genContextWithPassResult.clean();
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(FunctionDeclaration functionDeclarationAST, const GenContext &genContext)
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        if (mlirGenFunctionLikeDeclaration(functionDeclarationAST, genContext))
        {
            return mlir::success();
        }

        return mlir::failure();
    }

    mlir::Value mlirGen(FunctionExpression functionExpressionAST, const GenContext &genContext)
    {
        mlir_ts::FuncOp funcOp;

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.restoreInsertionPoint(functionBeginPoint);

            // provide name for it
            funcOp = mlirGenFunctionLikeDeclaration(functionExpressionAST, genContext);
            if (!funcOp)
            {
                return mlir::Value();
            }
        }

        auto funcSymbolRef = builder.create<mlir_ts::SymbolRefOp>(loc(functionExpressionAST), funcOp.getType(),
                                                                  mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
        return funcSymbolRef;
    }

    mlir::Value mlirGen(ArrowFunction arrowFunctionAST, const GenContext &genContext)
    {
        mlir_ts::FuncOp funcOp;

        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.restoreInsertionPoint(functionBeginPoint);

            // provide name for it
            funcOp = mlirGenFunctionLikeDeclaration(arrowFunctionAST, genContext);
            if (!funcOp)
            {
                return mlir::Value();
            }
        }

        auto funcSymbolRef = builder.create<mlir_ts::SymbolRefOp>(loc(arrowFunctionAST), funcOp.getType(),
                                                                  mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
        return funcSymbolRef;
    }

    mlir_ts::FuncOp mlirGenFunctionLikeDeclaration(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(functionLikeDeclarationBaseAST);

        auto funcOpWithFuncProto = mlirGenFunctionPrototype(functionLikeDeclarationBaseAST, genContext);

        auto &funcOp = std::get<0>(funcOpWithFuncProto);
        auto &funcProto = std::get<1>(funcOpWithFuncProto);
        auto result = std::get<2>(funcOpWithFuncProto);
        if (!result || !funcOp)
        {
            return funcOp;
        }

        auto funcGenContext = GenContext(genContext);
        funcGenContext.funcOp = funcOp;
        funcGenContext.passResult = nullptr;
        auto resultFromBody = mlirGenFunctionBody(functionLikeDeclarationBaseAST, funcOp, funcProto, funcGenContext);
        if (mlir::failed(resultFromBody))
        {
            return funcOp;
        }

        // set visibility index
        auto name = getNameWithoutNamespace(funcOp.getName());
        if (name != MAIN_ENTRY_NAME && !hasModifier(functionLikeDeclarationBaseAST, SyntaxKind::ExportKeyword))
        {
            funcOp.setPrivate();
        }

        if (!genContext.dummyRun)
        {
            theModule.push_back(funcOp);
        }

        if (!getFunctionMap().count(name))
        {
            getFunctionMap().insert({name, funcOp});

            LLVM_DEBUG(llvm::dbgs() << "reg. func: " << name << " type:" << funcOp.getType() << "\n";);
        }
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "re-process. func: " << name << " type:" << funcOp.getType() << "\n";);
        }

        return funcOp;
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
            declare(varDecl, entryOp.reference());
        }
        else
        {
            builder.create<mlir_ts::EntryOp>(location, mlir::Type());
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionExit(mlir::Location location, const GenContext &genContext)
    {
        builder.create<mlir_ts::ExitOp>(location);
        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionThisParam(mlir::Location loc, int &firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                                 mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        // register this if lambda function
        auto it = getCaptureVarsMap().find(funcProto->getName());
        if (it == getCaptureVarsMap().end())
        {
            return mlir::success();
        }

        firstIndex++;
        auto capturedVars = it->getValue();

        auto thisParam = arguments[firstIndex];
        auto thisRefType = thisParam.getType();

        auto thisParamVar = std::make_shared<VariableDeclarationDOM>(THIS_NAME, thisRefType, loc);

        declare(thisParamVar, thisParam);

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionParams(int firstIndex, FunctionPrototypeDOM::TypePtr funcProto,
                                              mlir::Block::BlockArgListType arguments, const GenContext &genContext)
    {
        auto index = firstIndex;
        for (const auto &param : funcProto->getArgs())
        {
            index++;
            mlir::Value paramValue;

            // alloc all args
            // process optional parameters
            if (param->getIsOptional() || param->hasInitValue())
            {
                // process init expression
                auto location = param->getLoc();

                auto optType = getOptionalType(param->getType());

                auto paramOptionalOp = builder.create<mlir_ts::ParamOptionalOp>(location, mlir_ts::RefType::get(optType), arguments[index]);

                paramValue = paramOptionalOp;

                if (param->hasInitValue())
                {
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

                    if (optType != defaultValue.getType())
                    {
                        defaultValue = builder.create<mlir_ts::CastOp>(location, optType, defaultValue);
                    }

                    builder.create<mlir_ts::ParamDefaultValueOp>(location, defaultValue);

                    builder.setInsertionPointAfter(paramOptionalOp);
                }
            }
            else
            {
                paramValue = builder.create<mlir_ts::ParamOp>(param->getLoc(), mlir_ts::RefType::get(param->getType()), arguments[index]);
            }

            if (paramValue)
            {
                // redefine variable
                param->setReadWriteAccess();
                declare(param, paramValue, true);
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionCaptures(FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
    {
        auto it = getCaptureVarsMap().find(funcProto->getName());
        if (it == getCaptureVarsMap().end())
        {
            return mlir::success();
        }

        auto capturedVars = it->getValue();

        NodeFactory nf(NodeFactoryFlags::None);

        // create variables
        for (auto &capturedVar : capturedVars)
        {
            auto varItem = capturedVar.getValue();
            auto variableInfo = varItem.second;
            auto name = variableInfo->getName();

            // load this.<var name>
            auto _this = nf.createIdentifier(stows(THIS_NAME));
            auto _name = nf.createIdentifier(stows(std::string(name)));
            auto _this_name = nf.createPropertyAccessExpression(_this, _name);
            auto thisVarRefValue = mlirGen(_this_name, genContext);
            auto variableRefType = mlir_ts::RefType::get(variableInfo->getType());

            auto capturedParam = std::make_shared<VariableDeclarationDOM>(name, variableRefType, variableInfo->getLoc());
            capturedParam->setReadWriteAccess();
            declare(capturedParam, thisVarRefValue);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGenFunctionBody(FunctionLikeDeclarationBase functionLikeDeclarationBaseAST, mlir_ts::FuncOp funcOp,
                                            FunctionPrototypeDOM::TypePtr funcProto, const GenContext &genContext)
    {
        auto location = loc(functionLikeDeclarationBaseAST);

        auto *blockPtr = funcOp.addEntryBlock();
        auto &entryBlock = *blockPtr;

        // process function params
        for (auto paramPairs : llvm::zip(funcProto->getArgs(), entryBlock.getArguments()))
        {
            if (failed(declare(std::get<0>(paramPairs), std::get<1>(paramPairs))))
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
        if (failed(mlirGenFunctionThisParam(location, firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

        // allocate function parameters as variable
        if (failed(mlirGenFunctionParams(firstIndex, funcProto, arguments, genContext)))
        {
            return mlir::failure();
        }

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

        auto typeInfo = getType(typeAssertionAST->type);
        auto exprValue = mlirGen(typeAssertionAST->expression, genContext);

        auto castedValue = builder.create<mlir_ts::CastOp>(location, typeInfo, exprValue);
        return castedValue;
    }

    mlir::Value mlirGen(AsExpression asExpressionAST, const GenContext &genContext)
    {
        auto location = loc(asExpressionAST);

        auto typeInfo = getType(asExpressionAST->type);
        auto exprValue = mlirGen(asExpressionAST->expression, genContext);

        auto castedValue = builder.create<mlir_ts::CastOp>(location, typeInfo, exprValue);
        return castedValue;
    }

    mlir::LogicalResult mlirGen(ReturnStatement returnStatementAST, const GenContext &genContext)
    {
        auto location = loc(returnStatementAST);
        if (auto expression = returnStatementAST->expression)
        {
            auto expressionValue = mlirGen(expression, genContext);
            return mlirGenReturnValue(location, expressionValue, genContext);
        }

        builder.create<mlir_ts::ReturnOp>(location);
        return mlir::success();
    }

    mlir::LogicalResult mlirGenReturnValue(mlir::Location location, mlir::Value expressionValue, const GenContext &genContext)
    {
        // empty return
        if (!expressionValue)
        {
            builder.create<mlir_ts::ReturnOp>(location);
            return mlir::success();
        }

        auto funcOp = const_cast<GenContext &>(genContext).funcOp;
        if (funcOp)
        {
            auto countResults = funcOp.getCallableResults().size();
            if (countResults > 0)
            {
                auto returnType = funcOp.getCallableResults().front();
                if (returnType != expressionValue.getType())
                {
                    auto castValue = builder.create<mlir_ts::CastOp>(location, returnType, expressionValue);
                    expressionValue = castValue;
                }
            }
        }

        // record return type if not provided
        if (genContext.passResult)
        {
            genContext.passResult->functionReturnType = expressionValue.getType();
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

        builder.create<mlir_ts::ReturnValOp>(location, expressionValue, retVarInfo.first);

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(IfStatement ifStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(ifStatementAST);

        auto hasElse = !!ifStatementAST->elseStatement;

        // condition
        auto condValue = mlirGen(ifStatementAST->expression, genContext);
        if (genContext.allowPartialResolve && !condValue)
        {
            return mlir::failure();
        }

        if (condValue.getType() != getBooleanType())
        {
            condValue = builder.create<mlir_ts::CastOp>(location, getBooleanType(), condValue);
        }

        auto ifOp = builder.create<mlir_ts::IfOp>(location, condValue, hasElse);

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
        mlirGen(forStatementAST->statement, genContext);
        builder.create<mlir_ts::ResultOp>(location);

        // increment
        builder.setInsertionPointToStart(&forOp.incr().front());
        mlirGen(forStatementAST->incrementor, genContext);
        builder.create<mlir_ts::ResultOp>(location);

        builder.setInsertionPointAfter(forOp);

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
        arrayVar->transformFlags |= TransformFlags::ForceConst;
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

    mlir::LogicalResult mlirGen(ForOfStatement forOfStatementAST, const GenContext &genContext)
    {
        SymbolTableScopeT varScope(symbolTable);

        auto location = loc(forOfStatementAST);

        NodeFactory nf(NodeFactoryFlags::None);

        // init
        NodeArray<VariableDeclaration> declarations;
        auto _i = nf.createIdentifier(S("_i_"));
        declarations.push_back(nf.createVariableDeclaration(_i, undefined, undefined, nf.createNumericLiteral(S("0"))));

        auto _a = nf.createIdentifier(S("_a_"));
        auto arrayVar = nf.createVariableDeclaration(_a, undefined, undefined, forOfStatementAST->expression);
        arrayVar->transformFlags |= TransformFlags::ForceConst;
        declarations.push_back(arrayVar);

        auto initVars = nf.createVariableDeclarationList(declarations, NodeFlags::Let);

        // condition
        auto cond = nf.createBinaryExpression(_i, nf.createToken(SyntaxKind::LessThanToken),
                                              nf.createPropertyAccessExpression(_a, nf.createIdentifier(S("length"))));

        // incr
        auto incr = nf.createPrefixUnaryExpression(nf.createToken(SyntaxKind::PlusPlusToken), _i);

        // block
        NodeArray<ts::Statement> statements;

        auto varDeclList = forOfStatementAST->initializer.as<VariableDeclarationList>();
        varDeclList->declarations.front()->initializer = nf.createElementAccessExpression(_a, _i);

        statements.push_back(nf.createVariableStatement(undefined, varDeclList));
        statements.push_back(forOfStatementAST->statement);
        auto block = nf.createBlock(statements);

        // final For statement
        auto forStatNode = nf.createForStatement(initVars, cond, incr, block);

        return mlirGen(forStatNode, genContext);
    }

    mlir::LogicalResult mlirGen(LabeledStatement labeledStatementAST, const GenContext &genContext)
    {
        auto location = loc(labeledStatementAST);

        label = MLIRHelper::getName(labeledStatementAST->label);
        auto res = mlirGen(labeledStatementAST->statement, genContext);

        return res;
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

    mlir::LogicalResult mlirGen(SwitchStatement switchStatementAST, const GenContext &genContext)
    {
        auto location = loc(switchStatementAST);

        auto switchValue = mlirGen(switchStatementAST->expression, genContext);

        auto switchOp = builder.create<mlir_ts::SwitchOp>(location, switchValue);

        // add merge block
        switchOp.addMergeBlock();
        auto *mergeBlock = switchOp.getMergeBlock();

        auto *lastBlock = mergeBlock;
        auto *lastConditionBlock = mergeBlock;

        auto &clauses = switchStatementAST->caseBlock->clauses;
        for (int index = clauses.size() - 1; index >= 0; index--)
        {
            auto caseBlock = clauses[index];
            auto statements = caseBlock->statements;
            // inline block
            if (statements.size() == 1)
            {
                auto firstStatement = statements.front();
                if ((SyntaxKind)firstStatement == SyntaxKind::Block)
                {
                    statements = statements.front().as<Block>()->statements;
                }
            }

            mlir::Block *caseBodyBlock = nullptr;
            mlir::Block *caseConditionBlock = nullptr;

            {
                mlir::OpBuilder::InsertionGuard guard(builder);
                caseBodyBlock = builder.createBlock(lastConditionBlock);

                auto hasBreak = false;
                for (auto statement : statements)
                {
                    if ((SyntaxKind)statement == SyntaxKind::BreakStatement)
                    {
                        hasBreak = true;
                        break;
                    }

                    mlirGen(statement, genContext);
                }

                // exit;
                builder.create<mlir::BranchOp>(location, hasBreak ? mergeBlock : lastBlock);

                lastBlock = caseBodyBlock;
            }

            switch ((SyntaxKind)caseBlock)
            {
            case SyntaxKind::CaseClause: {
                {

                    mlir::OpBuilder::InsertionGuard guard(builder);
                    caseConditionBlock = builder.createBlock(lastBlock);

                    auto caseValue = mlirGen(caseBlock.as<CaseClause>()->expression, genContext);

                    auto condition = builder.create<mlir_ts::LogicalBinaryOp>(
                        location, getBooleanType(), builder.getI32IntegerAttr((int)SyntaxKind::EqualsEqualsToken), switchValue, caseValue);

                    auto conditionI1 = builder.create<mlir_ts::CastOp>(location, builder.getI1Type(), condition);

                    builder.create<mlir::CondBranchOp>(location, conditionI1, caseBodyBlock, /*trueArguments=*/mlir::ValueRange{},
                                                       lastConditionBlock, /*falseArguments=*/mlir::ValueRange{});

                    lastConditionBlock = caseConditionBlock;
                }

                // create condition block
            }
            break;
            case SyntaxKind::DefaultClause: {
                lastConditionBlock = lastBlock;
            }
            break;
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(ThrowStatement throwStatementAST, const GenContext &genContext)
    {
        auto location = loc(throwStatementAST);

        auto exception = mlirGen(throwStatementAST->expression, genContext);

        builder.create<mlir_ts::ThrowOp>(location, exception);

        return mlir::success();

        // TODO: read about LLVM_ResumeOp,  maybe this is what you need (+LLVM_InvokeOp, LLVM_LandingpadOp)

        // TODO: PS, you can add param to each method to process return "exception info", and check every call for methods if they return
        // exception info

        /*
llvm.mlir.global external constant @_ZTIi() : !llvm.ptr<i8>
llvm.func @foo(!llvm.ptr<i8>)
llvm.func @bar(!llvm.ptr<i8>) -> !llvm.ptr<i8>
llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: @invokeLandingpad
llvm.func @invokeLandingpad() -> i32 attributes { personality = @__gxx_personality_v0 } {
// CHECK: %[[a1:[0-9]+]] = alloca i8
%0 = llvm.mlir.constant(0 : i32) : i32
%1 = llvm.mlir.constant("\01") : !llvm.array<1 x i8>
%2 = llvm.mlir.addressof @_ZTIi : !llvm.ptr<ptr<i8>>
%3 = llvm.bitcast %2 : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
%4 = llvm.mlir.null : !llvm.ptr<ptr<i8>>
%5 = llvm.mlir.constant(1 : i32) : i32
%6 = llvm.alloca %5 x i8 : (i32) -> !llvm.ptr<i8>
// CHECK: invoke void @foo(i8* %[[a1]])
// CHECK-NEXT: to label %[[normal:[0-9]+]] unwind label %[[unwind:[0-9]+]]
llvm.invoke @foo(%6) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>) -> ()

// CHECK: [[unwind]]:
^bb1:
// CHECK: %{{[0-9]+}} = landingpad { i8*, i32 }
// CHECK-NEXT:             catch i8** null
// CHECK-NEXT:             catch i8* bitcast (i8** @_ZTIi to i8*)
// CHECK-NEXT:             filter [1 x i8] c"\01"
%7 = llvm.landingpad (catch %4 : !llvm.ptr<ptr<i8>>) (catch %3 : !llvm.ptr<i8>) (filter %1 : !llvm.array<1 x i8>) : !llvm.struct<(ptr<i8>,
i32)>
// CHECK: br label %[[final:[0-9]+]]
llvm.br ^bb3

// CHECK: [[normal]]:
// CHECK-NEXT: ret i32 1
^bb2:	// 2 preds: ^bb0, ^bb3
llvm.return %5 : i32

// CHECK: [[final]]:
// CHECK-NEXT: %{{[0-9]+}} = invoke i8* @bar(i8* %[[a1]])
// CHECK-NEXT:          to label %[[normal]] unwind label %[[unwind]]
^bb3:	// pred: ^bb1
%8 = llvm.invoke @bar(%6) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
}

        */
    }

    mlir::LogicalResult mlirGen(TryStatement tryStatementAST, const GenContext &genContext)
    {
        auto location = loc(tryStatementAST);

        const_cast<GenContext &>(genContext).funcOp.personalityAttr(builder.getBoolAttr(true));

        auto tryOp = builder.create<mlir_ts::TryOp>(location);

        SmallVector<mlir::Type, 0> types;

        /*auto *body =*/builder.createBlock(&tryOp.body(), {}, types);
        /*auto *catches =*/builder.createBlock(&tryOp.catches(), {}, types);
        /*auto *finallyBlock =*/builder.createBlock(&tryOp.finallyBlock(), {}, types);

        // body
        builder.setInsertionPointToStart(&tryOp.body().front());
        auto result = mlirGen(tryStatementAST->tryBlock, genContext);
        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        // catches
        builder.setInsertionPointToStart(&tryOp.catches().front());
        // terminator
        builder.create<mlir_ts::ResultOp>(location);

        // finally
        builder.setInsertionPointToStart(&tryOp.finallyBlock().front());
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

        VALIDATE_EXPR(expressionValue, expression)

        auto boolValue = expressionValue;

        switch (opCode)
        {
        case SyntaxKind::ExclamationToken:

            if (expressionValue.getType() != getBooleanType())
            {
                boolValue = builder.create<mlir_ts::CastOp>(location, getBooleanType(), expressionValue);
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

        VALIDATE_EXPR(expressionValue, expression)

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
        auto condValue = mlirGen(conditionalExpressionAST->condition, genContext);
        if (condValue.getType() != getBooleanType())
        {
            condValue = builder.create<mlir_ts::CastOp>(location, getBooleanType(), condValue);
        }

        // detect value type
        mlir::Type resultType;
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            auto resultTrueTemp = mlirGen(conditionalExpressionAST->whenTrue, genContext);
            resultType = resultTrueTemp.getType();
        }

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, condValue, true);

        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        auto resultTrue = mlirGen(conditionalExpressionAST->whenTrue, genContext);
        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultTrue});

        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
        auto resultFalse = mlirGen(conditionalExpressionAST->whenFalse, genContext);
        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultFalse});

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
        auto resultType = leftExpressionValue.getType();

        auto condValue = builder.create<mlir_ts::CastOp>(location, getBooleanType(), leftExpressionValue);

        auto ifOp = builder.create<mlir_ts::IfOp>(location, mlir::TypeRange{resultType}, condValue, true);

        builder.setInsertionPointToStart(&ifOp.thenRegion().front());
        auto resultTrue = andOp ? mlirGen(rightExpression, genContext) : leftExpressionValue;
        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultTrue});

        builder.setInsertionPointToStart(&ifOp.elseRegion().front());
        auto resultFalse = andOp ? leftExpressionValue : mlirGen(rightExpression, genContext);

        // sync right part
        if (resultTrue.getType() != resultFalse.getType())
        {
            resultFalse = builder.create<mlir_ts::CastOp>(location, resultTrue.getType(), resultFalse);
        }

        builder.create<mlir_ts::ResultOp>(location, mlir::ValueRange{resultFalse});

        builder.setInsertionPointAfter(ifOp);

        return ifOp.getResult(0);
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

    mlir::Value mlirGenInstanceOfLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto result = mlirGen(binaryExpressionAST->left, genContext);
        auto resultType = result.getType();
        auto type = getTypeByTypeName(binaryExpressionAST->right, genContext);

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

    mlir::Value mlirGenSaveLogic(BinaryExpression binaryExpressionAST, const GenContext &genContext)
    {
        auto location = loc(binaryExpressionAST);

        auto leftExpression = binaryExpressionAST->left;
        auto rightExpression = binaryExpressionAST->right;

        auto rightExpressionValue = mlirGen(rightExpression, genContext);
        auto leftExpressionValue = mlirGen(leftExpression, genContext);

        VALIDATE_EXPR(rightExpressionValue, rightExpression)
        VALIDATE_EXPR(leftExpressionValue, leftExpression)

        auto leftExpressionValueBeforeCast = leftExpressionValue;

        if (leftExpressionValue.getType() != rightExpressionValue.getType())
        {
            if (rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
            {
                rightExpressionValue = builder.create<mlir_ts::CastOp>(loc(rightExpression), getStringType(), rightExpressionValue);
            }
        }

        auto result = rightExpressionValue;

        // saving
        if (leftExpressionValueBeforeCast.getType() != result.getType())
        {
            result = builder.create<mlir_ts::CastOp>(loc(leftExpression), leftExpressionValueBeforeCast.getType(), result);
        }

        // TODO: finish it for field access, review CodeLogicHelper.saveResult
        if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(leftExpressionValueBeforeCast.getDefiningOp()))
        {
            // TODO: when saving const array into variable we need to allocate space and copy array as we need to have writable array
            builder.create<mlir_ts::StoreOp>(location, result, loadOp.reference());
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        return result;
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

        VALIDATE_EXPR(rightExpressionValue, rightExpression)
        VALIDATE_EXPR(leftExpressionValue, leftExpression)

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

        if (leftExpressionValue.getType() != rightExpressionValue.getType())
        {
            // TODO: temporary hack
            if (leftExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
            {
                leftExpressionValue = builder.create<mlir_ts::CastOp>(loc(leftExpression), getStringType(), leftExpressionValue);
            }

            if (rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::CharType>())
            {
                rightExpressionValue = builder.create<mlir_ts::CastOp>(loc(rightExpression), getStringType(), rightExpressionValue);
            }

            // end todo

            if (!MLIRLogicHelper::isLogicOp(opCode))
            {
                // cast from optional<T> type
                if (auto leftOptType = leftExpressionValue.getType().dyn_cast_or_null<mlir_ts::OptionalType>())
                {
                    leftExpressionValue =
                        builder.create<mlir_ts::CastOp>(loc(leftExpression), leftOptType.getElementType(), leftExpressionValue);
                }

                if (auto rightOptType = rightExpressionValue.getType().dyn_cast_or_null<mlir_ts::OptionalType>())
                {
                    rightExpressionValue =
                        builder.create<mlir_ts::CastOp>(loc(rightExpression), rightOptType.getElementType(), rightExpressionValue);
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
                leftExpressionValue = builder.create<mlir_ts::CastOp>(loc(leftExpression), builder.getI32Type(), leftExpressionValue);
            }

            if (rightExpressionValue.getType() != builder.getI32Type())
            {
                rightExpressionValue = builder.create<mlir_ts::CastOp>(loc(rightExpression), builder.getI32Type(), rightExpressionValue);
            }

            break;
        case SyntaxKind::SlashToken:
        case SyntaxKind::PercentToken:
        case SyntaxKind::AsteriskAsteriskToken:

            if (leftExpressionValue.getType() != builder.getF32Type())
            {
                leftExpressionValue = builder.create<mlir_ts::CastOp>(loc(leftExpression), builder.getF32Type(), leftExpressionValue);
            }

            if (rightExpressionValue.getType() != builder.getF32Type())
            {
                rightExpressionValue = builder.create<mlir_ts::CastOp>(loc(rightExpression), builder.getF32Type(), rightExpressionValue);
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
                auto hasF32 =
                    leftExpressionValue.getType() == builder.getF32Type() || rightExpressionValue.getType() == builder.getF32Type();
                if (hasF32)
                {
                    if (leftExpressionValue.getType() != builder.getF32Type())
                    {
                        leftExpressionValue =
                            builder.create<mlir_ts::CastOp>(loc(leftExpression), builder.getF32Type(), leftExpressionValue);
                    }

                    if (rightExpressionValue.getType() != builder.getF32Type())
                    {
                        rightExpressionValue =
                            builder.create<mlir_ts::CastOp>(loc(rightExpression), builder.getF32Type(), rightExpressionValue);
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
                            leftExpressionValue =
                                builder.create<mlir_ts::CastOp>(loc(leftExpression), builder.getI32Type(), leftExpressionValue);
                        }

                        if (rightExpressionValue.getType() != builder.getI32Type())
                        {
                            rightExpressionValue =
                                builder.create<mlir_ts::CastOp>(loc(rightExpression), builder.getI32Type(), rightExpressionValue);
                        }
                    }
                }
            }

            break;
        default:
            if (leftExpressionValue.getType() != rightExpressionValue.getType())
            {
                rightExpressionValue =
                    builder.create<mlir_ts::CastOp>(loc(rightExpression), leftExpressionValue.getType(), rightExpressionValue);
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
                result = builder.create<mlir_ts::CastOp>(loc(leftExpression), leftExpressionValueBeforeCast.getType(), result);
            }

            // TODO: finish it for field access, review CodeLogicHelper.saveResult
            if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(leftExpressionValueBeforeCast.getDefiningOp()))
            {
                // TODO: when saving const array into variable we need to allocate space and copy array as we need to have writable array
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

        VALIDATE_EXPR(expressionValue, expression)

        auto name = MLIRHelper::getName(qualifiedName->right);

        mlir::Value value;
        if (!expressionValue.getType() || expressionValue.getType() == mlir::NoneType::get(builder.getContext()))
        {
            if (auto namespaceRef = dyn_cast_or_null<mlir_ts::NamespaceRefOp>(expressionValue.getDefiningOp()))
            {
                // todo resolve namespace
                auto namespaceInfo = getNamespaceByFullName(namespaceRef.identifier());

                assert(namespaceInfo);

                auto saveNamespace = currentNamespace;
                currentNamespace = namespaceInfo;

                value = mlirGen(location, name, genContext);

                currentNamespace = saveNamespace;
            }

            return value;
        }

        emitError(location, "Can't resolve qualified name");

        llvm_unreachable("not implemented");
    }

    mlir::Value mlirGen(PropertyAccessExpression propertyAccessExpression, const GenContext &genContext)
    {
        auto location = loc(propertyAccessExpression);

        auto expression = propertyAccessExpression->expression.as<Expression>();
        auto expressionValue = mlirGen(expression, genContext);

        VALIDATE_EXPR(expressionValue, expression)

        auto name = MLIRHelper::getName(propertyAccessExpression->name);

        mlir::Value value;

        if (!expressionValue.getType() || expressionValue.getType() == mlir::NoneType::get(builder.getContext()))
        {
            if (auto namespaceRef = dyn_cast_or_null<mlir_ts::NamespaceRefOp>(expressionValue.getDefiningOp()))
            {
                // todo resolve namespace
                auto namespaceInfo = getNamespaceByFullName(namespaceRef.identifier());

                assert(namespaceInfo);

                auto saveNamespace = currentNamespace;
                currentNamespace = namespaceInfo;

                value = mlirGen(location, name, genContext);

                currentNamespace = saveNamespace;
            }

            return value;
        }

        MLIRPropertyAccessCodeLogic cl(builder, location, expressionValue, name);

        TypeSwitch<mlir::Type>(expressionValue.getType())
            .Case<mlir_ts::EnumType>([&](auto enumType) { value = cl.Enum(enumType); })
            .Case<mlir_ts::ConstTupleType>([&](auto tupleType) { value = cl.Tuple(tupleType); })
            .Case<mlir_ts::TupleType>([&](auto tupleType) { value = cl.Tuple(tupleType); })
            .Case<mlir_ts::BooleanType>([&](auto intType) { value = cl.Bool(intType); })
            .Case<mlir::IntegerType>([&](auto intType) { value = cl.Int(intType); })
            .Case<mlir::FloatType>([&](auto intType) { value = cl.Float(intType); })
            .Case<mlir_ts::StringType>([&](auto stringType) { value = cl.String(stringType); })
            .Case<mlir_ts::ConstArrayType>([&](auto arrayType) { value = cl.Array(arrayType); })
            .Case<mlir_ts::ArrayType>([&](auto arrayType) { value = cl.Array(arrayType); })
            .Case<mlir_ts::RefType>([&](auto refType) { value = cl.Ref(refType); })
            .Case<mlir_ts::ClassType>([&](auto classType) { value = cl.Class(expressionValue, classType); })
            .Default([](auto type) { llvm_unreachable("not implemented"); });

        if (value)
        {
            return value;
        }

        emitError(location, "Can't resolve property name");

        llvm_unreachable("not implemented");
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

        auto elemRef = builder.create<mlir_ts::ElementRefOp>(location, mlir_ts::RefType::get(elementType), expression, argumentExpression);
        return builder.create<mlir_ts::LoadOp>(location, elementType, elemRef);
    }

    mlir::Value mlirGen(CallExpression callExpression, const GenContext &genContext)
    {
        auto location = loc(callExpression);

        // get function ref.
        auto funcRefValue = mlirGen(callExpression->expression.as<Expression>(), genContext);
        if (!funcRefValue && genContext.allowPartialResolve)
        {
            return mlir::Value();
        }

        auto attrName = StringRef(IDENTIFIER_ATTR_NAME);
        auto definingOp = funcRefValue.getDefiningOp();
        if (funcRefValue.getType() == mlir::NoneType::get(builder.getContext()) &&
            definingOp->hasAttrOfType<mlir::FlatSymbolRefAttr>(attrName))
        {
            // TODO: when you resolve names such as "print", "parseInt" should return names in mlirGen(Identifier)
            auto calleeName = definingOp->getAttrOfType<mlir::FlatSymbolRefAttr>(attrName);
            auto functionName = calleeName.getValue();
            auto argumentsContext = callExpression->arguments;

            // resolve function
            MLIRCustomMethods cm(builder, location);

            SmallVector<mlir::Value, 4> operands;
            if (mlir::failed(mlirGen(argumentsContext, operands, genContext)))
            {
                if (!genContext.dummyRun)
                {
                    emitError(location) << "Call Method: can't resolve values of all parameters";
                }

                return mlir::Value();
            }

            return cm.callMethod(functionName, operands, genContext.allowPartialResolve);
        }

        mlir::Value value;
        TypeSwitch<mlir::Type>(funcRefValue.getType())
            .Case<mlir::FunctionType>([&](auto calledFuncType) {
                SmallVector<mlir::Value, 4> operands;
                if (mlir::failed(mlirGenCallOperands(location, calledFuncType, callExpression->arguments, operands, genContext)))
                {
                    if (!genContext.dummyRun)
                    {
                        emitError(location) << "Call Method: can't resolve values of all parameters";
                    }
                }
                else
                {
                    // default call by name
                    auto callIndirectOp = builder.create<mlir_ts::CallIndirectOp>(location, funcRefValue, operands);

                    if (calledFuncType.getNumResults() > 0)
                    {
                        value = callIndirectOp.getResult(0);
                    }
                }
            })
            .Default([&](auto type) {
                // it is not function, so just return value as maybe resolved earlier like in case "<number>.ToString()"
                value = funcRefValue;
            });

        if (value)
        {
            return value;
        }

        return nullptr;
    }

    mlir::LogicalResult mlirGenCallOperands(mlir::Location location, mlir::FunctionType calledFuncType,
                                            NodeArray<Expression> argumentsContext, SmallVector<mlir::Value, 4> &operands,
                                            const GenContext &genContext)
    {
        auto opArgsCount = std::distance(argumentsContext.begin(), argumentsContext.end());
        auto funcArgsCount = calledFuncType.getNumInputs();

        if (mlir::failed(mlirGen(argumentsContext, operands, calledFuncType, genContext)))
        {
            return mlir::failure();
        }

        if (funcArgsCount > opArgsCount)
        {
            // -1 to exclude count params
            for (auto i = (size_t)opArgsCount; i < funcArgsCount; i++)
            {
                operands.push_back(builder.create<mlir_ts::UndefOp>(location, calledFuncType.getInput(i)));
            }
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(NodeArray<Expression> arguments, SmallVector<mlir::Value, 4> &operands, const GenContext &genContext)
    {
        for (auto expression : arguments)
        {
            auto value = mlirGen(expression, genContext);
            if (!value)
            {
                return mlir::failure();
            }

            operands.push_back(value);
        }

        return mlir::success();
    }

    mlir::LogicalResult mlirGen(NodeArray<Expression> arguments, SmallVector<mlir::Value, 4> &operands, mlir::FunctionType funcType,
                                const GenContext &genContext)
    {
        auto i = 0;
        for (auto expression : arguments)
        {
            auto value = mlirGen(expression, genContext);
            if (value.getType() != funcType.getInput(i))
            {
                auto castValue = builder.create<mlir_ts::CastOp>(loc(expression), funcType.getInput(i), value);
                operands.push_back(castValue);
            }
            else
            {
                operands.push_back(value);
            }

            i++;
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
            type = getTypeByTypeName(typeExpression, genContext);
            auto resultType = type;
            if (mth.isValueType(type))
            {
                resultType = getValueRefType(type);
            }

            auto newOp = builder.create<mlir_ts::NewOp>(location, resultType);
            return newOp;
        }
        else if (typeExpression == SyntaxKind::ElementAccessExpression)
        {
            auto elementAccessExpression = typeExpression.as<ElementAccessExpression>();
            typeExpression = elementAccessExpression->expression;
            type = getTypeByTypeName(typeExpression, genContext);
            auto count = mlirGen(elementAccessExpression->argumentExpression, genContext);

            if (count.getType() != builder.getI32Type())
            {
                count = builder.create<mlir_ts::CastOp>(location, builder.getI32Type(), count);
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
                expr = builder.create<mlir_ts::CastOp>(location, mlir_ts::RefType::get(arrayType.getElementType()), expr);
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
        auto result = mlirGen(typeOfExpression->expression, genContext);
        auto type = result.getType();
        if (type.isIntOrIndexOrFloat() && !type.isIntOrIndex())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("number")));
            return typeOfValue;
        }

        if (type == getBooleanType())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getBooleanType(), getStringAttr(std::string("boolean")));
            return typeOfValue;
        }

        if (type == getStringType())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("string")));
            return typeOfValue;
        }

        if (type.dyn_cast_or_null<mlir_ts::ArrayType>())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("array")));
            return typeOfValue;
        }

        if (type.dyn_cast_or_null<mlir::FunctionType>())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("function")));
            return typeOfValue;
        }

        if (type == getAnyType())
        {
            auto typeOfValue =
                builder.create<mlir_ts::ConstantOp>(loc(typeOfExpression), getStringType(), getStringAttr(std::string("object")));
            return typeOfValue;
        }

        llvm_unreachable("not implemented");
    }

    mlir::Value mlirGen(TemplateLiteralLikeNode templateExpressionAST, const GenContext &genContext)
    {
        auto location = loc(templateExpressionAST);

        auto stringType = getStringType();
        SmallVector<mlir::Value, 4> strs;

        auto text = wstos(templateExpressionAST->head->rawText);
        auto head = builder.create<mlir_ts::ConstantOp>(location, stringType, getStringAttr(text));

        // first string
        strs.push_back(head);
        for (auto span : templateExpressionAST->templateSpans)
        {
            auto expression = span->expression;
            auto exprValue = mlirGen(expression, genContext);

            VALIDATE_EXPR(exprValue, expression)

            if (exprValue.getType() != stringType)
            {
                exprValue = builder.create<mlir_ts::CastOp>(location, stringType, exprValue);
            }

            // expr value
            strs.push_back(exprValue);

            auto spanText = wstos(span->literal->rawText);
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

        auto text = wstos(templateExpressionAST->head->rawText);

        // first string
        strs.push_back(getStringAttr(text));
        for (auto span : templateExpressionAST->templateSpans)
        {
            // expr value
            auto expression = span->expression;
            auto exprValue = mlirGen(expression, genContext);

            VALIDATE_EXPR(exprValue, expression)

            vals.push_back(exprValue);

            auto spanText = wstos(span->literal->rawText);
            // text
            strs.push_back(getStringAttr(spanText));
        }

        // tag method
        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), strs);
        auto constStringArray = builder.create<mlir_ts::ConstantOp>(location, getConstArrayType(getStringType(), strs.size()), arrayAttr);

        auto strArrayValue = builder.create<mlir_ts::CastOp>(location, getArrayType(getStringType()), constStringArray);

        vals.insert(vals.begin(), strArrayValue);

        auto callee = mlirGen(taggedTemplateExpressionAST->tag, genContext);

        // cast all params if needed
        auto funcType = callee.getType().cast<mlir::FunctionType>();

        SmallVector<mlir::Value, 4> operands;

        auto i = 0;
        for (auto value : vals)
        {
            if (value.getType() != funcType.getInput(i))
            {
                auto castValue = builder.create<mlir_ts::CastOp>(value.getLoc(), funcType.getInput(i), value);
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
        return builder.create<mlir_ts::NullOp>(loc(nullLiteral), getAnyType());
    }

    mlir::Value mlirGen(TrueLiteral trueLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::ConstantOp>(loc(trueLiteral), getBooleanType(), mlir::BoolAttr::get(theModule.getContext(), true));
    }

    mlir::Value mlirGen(FalseLiteral falseLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::ConstantOp>(loc(falseLiteral), getBooleanType(), mlir::BoolAttr::get(theModule.getContext(), false));
    }

    mlir::Value mlirGen(NumericLiteral numericLiteral, const GenContext &genContext)
    {
        if (numericLiteral->text.find(S(".")) == string::npos)
        {
            return builder.create<mlir_ts::ConstantOp>(loc(numericLiteral), builder.getI32Type(),
                                                       builder.getI32IntegerAttr(to_unsigned_integer(numericLiteral->text)));
        }

        return builder.create<mlir_ts::ConstantOp>(loc(numericLiteral), builder.getF32Type(),
                                                   builder.getF32FloatAttr(to_float(numericLiteral->text)));
    }

    mlir::Value mlirGen(BigIntLiteral bigIntLiteral, const GenContext &genContext)
    {
        return builder.create<mlir_ts::ConstantOp>(loc(bigIntLiteral), builder.getI64Type(),
                                                   builder.getI64IntegerAttr(to_bignumber(bigIntLiteral->text)));
    }

    mlir::Value mlirGen(ts::StringLiteral stringLiteral, const GenContext &genContext)
    {
        auto text = wstos(stringLiteral->text);

        return builder.create<mlir_ts::ConstantOp>(loc(stringLiteral), getStringType(), getStringAttr(text));
    }

    mlir::Value mlirGen(ts::NoSubstitutionTemplateLiteral noSubstitutionTemplateLiteral, const GenContext &genContext)
    {
        auto text = wstos(noSubstitutionTemplateLiteral->text);

        return builder.create<mlir_ts::ConstantOp>(loc(noSubstitutionTemplateLiteral), getStringType(), getStringAttr(text));
    }

    mlir::Value mlirGen(ts::ArrayLiteralExpression arrayLiteral, const GenContext &genContext)
    {
        auto location = loc(arrayLiteral);

        MLIRTypeHelper mth(builder.getContext());

        // first value
        auto isTuple = false;
        mlir::Type elementType;
        SmallVector<mlir::Type> types;
        SmallVector<mlir::Attribute> values;
        for (auto &item : arrayLiteral->elements)
        {
            auto itemValue = mlirGen(item, genContext);
            if (!itemValue)
            {
                // omitted expression
                continue;
            }

            auto constOp = itemValue.getDefiningOp<mlir_ts::ConstantOp>();
            if (!constOp)
            {
                emitError(location, "Array literal should contains constant values only");
                llvm_unreachable("array literal is not implemented(1)");
                continue;
            }

            bool copyRequired;
            auto type = mth.convertConstTypeToType(constOp.getType(), copyRequired);

            values.push_back(constOp.valueAttr());
            types.push_back(type);
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

        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), values);
        if (isTuple)
        {
            SmallVector<mlir_ts::FieldInfo> fieldInfos;
            for (auto type : types)
            {
                fieldInfos.push_back({mlir::Attribute(), type});
            }

            return builder.create<mlir_ts::ConstantOp>(loc(arrayLiteral), getConstTupleType(fieldInfos), arrayAttr);
        }

        return builder.create<mlir_ts::ConstantOp>(loc(arrayLiteral), getConstArrayType(elementType, values.size()), arrayAttr);
    }

    mlir::Value mlirGen(ts::ObjectLiteralExpression objectLiteral, const GenContext &genContext)
    {
        MLIRCodeLogic mcl(builder);
        // first value
        SmallVector<mlir::Type> types;
        SmallVector<mlir_ts::FieldInfo> fieldInfos;
        SmallVector<mlir::Attribute> values;
        for (auto &item : objectLiteral->properties)
        {
            mlir::Value itemValue;
            mlir::Attribute fieldId;
            if (item == SyntaxKind::PropertyAssignment)
            {
                auto propertyAssignment = item.as<PropertyAssignment>();
                itemValue = mlirGen(propertyAssignment->initializer, genContext);
                auto name = MLIRHelper::getName(propertyAssignment->name);
                if (name.empty())
                {
                    auto value = mlirGen(propertyAssignment->name.as<Expression>(), genContext);
                    fieldId = mcl.ExtractAttr(value);
                }
                else
                {
                    auto namePtr = StringRef(name).copy(stringAllocator);
                    fieldId = mcl.TupleFieldName(namePtr);
                }
            }
            else if (item == SyntaxKind::ShorthandPropertyAssignment)
            {
                auto shorthandPropertyAssignment = item.as<ShorthandPropertyAssignment>();
                itemValue = mlirGen(shorthandPropertyAssignment->name.as<Expression>(), genContext);
                auto name = MLIRHelper::getName(shorthandPropertyAssignment->name);
                auto namePtr = StringRef(name).copy(stringAllocator);
                fieldId = mcl.TupleFieldName(namePtr);
            }
            else
            {
                llvm_unreachable("object literal is not implemented(1)");
            }

            mlir::Type type;
            mlir::Attribute value;
            if (auto constOp = dyn_cast_or_null<mlir_ts::ConstantOp>(itemValue.getDefiningOp()))
            {
                value = constOp.valueAttr();
                type = constOp.getType();
            }
            else if (auto symRefOp = dyn_cast_or_null<mlir_ts::SymbolRefOp>(itemValue.getDefiningOp()))
            {
                value = symRefOp.identifierAttr();
                type = symRefOp.getType();
            }
            else
            {
                llvm_unreachable("object literal is not implemented(1), must be const object or global symbol");
                continue;
            }

            values.push_back(value);
            types.push_back(type);
            fieldInfos.push_back({fieldId, type});
        }

        auto arrayAttr = mlir::ArrayAttr::get(builder.getContext(), values);
        return builder.create<mlir_ts::ConstantOp>(loc(objectLiteral), getConstTupleType(fieldInfos), arrayAttr);
    }

    mlir::Value mlirGen(Identifier identifier, const GenContext &genContext)
    {
        auto location = loc(identifier);

        // resolve name
        auto name = MLIRHelper::getName(identifier);

        return mlirGen(location, name, genContext);
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
            if (genContext.funcOp)
            {
                auto funcRegion = const_cast<GenContext &>(genContext).funcOp.getCallableRegion();
                isOuterVar = !funcRegion->isAncestor(valueRegion);
            }

            // auto isOuterFunctionScope = value.second->getFuncOp() != genContext.funcOp;
            //
            // LLVM_DEBUG(llvm::dbgs() << "isOuterFunctionScope: " << (isOuterFunctionScope ? "true" : "false")
            //                         << ", isOuterVar: " << (isOuterVar ? "true" : "false") << " name: " << name << "\n");

            if (isOuterVar && genContext.passResult)
            {
                LLVM_DEBUG(llvm::dbgs() << "outer var name: " << name << " type: " << value.first.getType() << " value: " << value.first
                                        << "\n");

                genContext.passResult->outerVariables.insert({value.second->getName(), value});
            }

            // end of logic: outer vars

            if (!value.second->getReadWriteAccess())
            {
                return value.first;
            }

            // load value if memref
            auto valueType = value.first.getType().cast<mlir_ts::RefType>().getElementType();
            return builder.create<mlir_ts::LoadOp>(value.first.getLoc(), valueType, value.first);
        }

        return mlir::Value();
    }

    mlir::Value resolveIdentifierInNamespace(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // resolving function
        auto fn = getFunctionMap().find(name);
        if (fn != getFunctionMap().end())
        {
            auto funcOp = fn->getValue();
            auto effectiveFuncType = funcOp.getType();
            // check if required capture of vars
            auto captureVars = getCaptureVarsMap().find(name);
            if (captureVars != getCaptureVarsMap().end())
            {
                auto funcType = effectiveFuncType;
                auto newFuncType = builder.getFunctionType(funcType.getInputs().slice(1), funcType.getResults());
                effectiveFuncType = newFuncType;

                auto funcSymbolOp =
                    builder.create<mlir_ts::SymbolRefOp>(location, funcType, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));

                MLIRCodeLogic mcl(builder);
                SmallVector<mlir::Value> capturedValues;
                for (auto &item : captureVars->getValue())
                {
                    auto varValue = mlirGen(location, item.first(), genContext);
                    auto refValue = mcl.GetReferenceOfLoadOp(varValue);
                    assert(refValue);
                    capturedValues.push_back(refValue);
                }

                auto captured = builder.create<mlir_ts::CaptureOp>(location, funcType.getInput(0), capturedValues);
                return builder.create<mlir_ts::TrampolineOp>(location, effectiveFuncType, funcSymbolOp, captured);
            }

            auto symbOp = builder.create<mlir_ts::SymbolRefOp>(location, effectiveFuncType,
                                                               mlir::FlatSymbolRefAttr::get(builder.getContext(), funcOp.getName()));
            return symbOp;
        }

        if (getGlobalsMap().count(name))
        {
            auto value = getGlobalsMap().lookup(name);
            if (!value->getReadWriteAccess() && value->getType().isa<mlir_ts::StringType>())
            {
                // load address of const object in global
                return builder.create<mlir_ts::AddressOfConstStringOp>(location, value->getType(), value->getName());
            }
            else
            {
                auto address = builder.create<mlir_ts::AddressOfOp>(location, mlir_ts::RefType::get(value->getType()), value->getName());
                return builder.create<mlir_ts::LoadOp>(location, value->getType(), address);
            }
        }

        // check if we have enum
        if (getEnumsMap().count(name))
        {
            auto enumTypeInfo = getEnumsMap().lookup(name);
            return builder.create<mlir_ts::ConstantOp>(location, getEnumType(enumTypeInfo.first), enumTypeInfo.second);
        }

        if (getClassesMap().count(name))
        {
            auto classInfo = getClassesMap().lookup(name);
            return builder.create<mlir_ts::ClassRefOp>(location, classInfo->storageType,
                                                       mlir::FlatSymbolRefAttr::get(builder.getContext(), classInfo->fullName));
        }

        if (getTypeAliasMap().count(name))
        {
            auto typeAliasInfo = getTypeAliasMap().lookup(name);
            return builder.create<mlir_ts::TypeRefOp>(location, typeAliasInfo);
        }

        if (getNamespaceMap().count(name))
        {
            auto namespaceInfo = getNamespaceMap().lookup(name);
            return builder.create<mlir_ts::NamespaceRefOp>(location,
                                                           mlir::FlatSymbolRefAttr::get(builder.getContext(), namespaceInfo->fullName));
        }

        if (getImportEqualsMap().count(name))
        {
            auto fullName = getImportEqualsMap().lookup(name);
            return builder.create<mlir_ts::NamespaceRefOp>(location, mlir::FlatSymbolRefAttr::get(builder.getContext(), fullName));
        }

        return mlir::Value();
    }

    mlir::Value resolveIdentifier(mlir::Location location, StringRef name, const GenContext &genContext)
    {
        // built in types
        if (name == "undefined")
        {
            return getUndefined(location);
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

        // search in root namespace
        auto saveNamespace = currentNamespace;
        currentNamespace = rootNamespace;
        value = resolveIdentifierInNamespace(location, name, genContext);
        currentNamespace = saveNamespace;
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
        return builder.create<mlir_ts::SymbolRefOp>(location, mlir::FlatSymbolRefAttr::get(builder.getContext(), name));
    }

    mlir::LogicalResult mlirGen(TypeAliasDeclaration typeAliasDeclarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(typeAliasDeclarationAST->name);
        if (!name.empty())
        {
            auto type = getType(typeAliasDeclarationAST->type);
            getTypeAliasMap().insert({name, type});
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
        }
        else
        {
            llvm_unreachable("not implemented");
        }

        return mlir::failure();
    }

    mlir::LogicalResult mlirGen(EnumDeclaration enumDeclarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(enumDeclarationAST->name);
        if (name.empty())
        {
            llvm_unreachable("not implemented");
            return mlir::failure();
        }

        auto namePtr = StringRef(name).copy(stringAllocator);

        SmallVector<mlir::NamedAttribute> enumValues;
        int64_t index = 0;
        auto activeBits = 0;
        for (auto enumMember : enumDeclarationAST->members)
        {
            auto memberName = MLIRHelper::getName(enumMember->name);
            if (memberName.empty())
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

            enumValues.push_back({mlir::Identifier::get(memberName, builder.getContext()), enumValueAttr});
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

    mlir::LogicalResult mlirGen(ClassLikeDeclaration classDeclarationAST, const GenContext &genContext)
    {
        auto name = MLIRHelper::getName(classDeclarationAST->name);
        if (name.empty())
        {
            llvm_unreachable("not implemented");
            return mlir::failure();
        }

        auto namePtr = StringRef(name).copy(stringAllocator);
        auto fullNamePtr = getFullNamespaceName(namePtr);

        // register class
        auto newClassPtr = std::make_shared<ClassInfo>();
        newClassPtr->name = namePtr;
        newClassPtr->fullName = fullNamePtr;
        getClassesMap().insert({namePtr, newClassPtr});

        // read class info
        MLIRCodeLogic mcl(builder);
        // first value
        SmallVector<mlir::Type> types;
        SmallVector<mlir_ts::FieldInfo> fieldInfos;
        SmallVector<mlir::Attribute> values;
        for (auto &classMember : classDeclarationAST->members)
        {
            mlir::Value initValue;
            mlir::Attribute fieldId;
            mlir::Type type;
            auto isStatic = false;

            if (classMember == SyntaxKind::PropertyDeclaration)
            {
                auto propertyDeclaration = classMember.as<PropertyDeclaration>();

                auto memberName = MLIRHelper::getName(propertyDeclaration->name);
                if (memberName.empty())
                {
                    llvm_unreachable("not implemented");
                    return mlir::failure();
                }

                auto namePtr = StringRef(memberName).copy(stringAllocator);
                fieldId = mcl.TupleFieldName(namePtr);

                type = getType(propertyDeclaration->type);

                isStatic = hasModifier(propertyDeclaration, SyntaxKind::StaticKeyword);
            }

            if (!isStatic)
            {
                fieldInfos.push_back({fieldId, type});
            }
        }

        auto classType = getClassType(getTupleType(fieldInfos));
        newClassPtr->storageType = classType;

        return mlir::success();
    }

    mlir::Type getType(Node typeReferenceAST)
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
            return getFunctionType(typeReferenceAST.as<FunctionTypeNode>());
        }
        else if (kind == SyntaxKind::TupleType)
        {
            return getTupleType(typeReferenceAST.as<TupleTypeNode>());
        }
        else if (kind == SyntaxKind::TypeLiteral)
        {
            return getTupleType(typeReferenceAST.as<TypeLiteralNode>());
        }
        else if (kind == SyntaxKind::ArrayType)
        {
            return getArrayType(typeReferenceAST.as<ArrayTypeNode>());
        }
        else if (kind == SyntaxKind::UnionType)
        {
            return getUnionType(typeReferenceAST.as<UnionTypeNode>());
        }
        else if (kind == SyntaxKind::IntersectionType)
        {
            return getIntersectionType(typeReferenceAST.as<IntersectionTypeNode>());
        }
        else if (kind == SyntaxKind::ParenthesizedType)
        {
            return getParenthesizedType(typeReferenceAST.as<ParenthesizedTypeNode>());
        }
        else if (kind == SyntaxKind::LiteralType)
        {
            return getLiteralType(typeReferenceAST.as<LiteralTypeNode>());
        }
        else if (kind == SyntaxKind::TypeReference)
        {
            GenContext genContext;
            return getTypeByTypeReference(typeReferenceAST.as<TypeReferenceNode>(), genContext);
        }
        else if (kind == SyntaxKind::AnyKeyword)
        {
            return getAnyType();
        }

        llvm_unreachable("not implemented type declaration");
        // return getAnyType();
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

            return type;
        }

        llvm_unreachable("not implemented");
    }

    mlir::Type getTypeByTypeReference(TypeReferenceNode typeReferenceAST, const GenContext &genContext)
    {
        return getTypeByTypeName(typeReferenceAST->typeName, genContext);
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

    mlir_ts::EnumType getEnumType()
    {
        return getEnumType(builder.getI32Type());
    }

    mlir_ts::EnumType getEnumType(mlir::Type elementType)
    {
        return mlir_ts::EnumType::get(elementType);
    }

    mlir_ts::ClassType getClassType(mlir::Type storageType)
    {
        return mlir_ts::ClassType::get(storageType);
    }

    mlir_ts::ConstArrayType getConstArrayType(ArrayTypeNode arrayTypeAST, unsigned size)
    {
        auto type = getType(arrayTypeAST->elementType);
        return getConstArrayType(type, size);
    }

    mlir_ts::ConstArrayType getConstArrayType(mlir::Type elementType, unsigned size)
    {
        assert(elementType);
        return mlir_ts::ConstArrayType::get(elementType, size);
    }

    mlir_ts::ArrayType getArrayType(ArrayTypeNode arrayTypeAST)
    {
        auto type = getType(arrayTypeAST->elementType);
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

    mlir::Value getUndefined(mlir::Location location)
    {
        return builder.create<mlir_ts::UndefOp>(location, getOptionalType(getUndefPlaceHolderType()));
    }

    void getTupleFieldInfo(TupleTypeNode tupleType, mlir::SmallVector<mlir_ts::FieldInfo> &types)
    {
        MLIRCodeLogic mcl(builder);
        mlir::Attribute attrVal;
        for (auto typeItem : tupleType->elements)
        {
            if (typeItem == SyntaxKind::NamedTupleMember)
            {
                auto namedTupleMember = typeItem.as<NamedTupleMember>();
                auto namePtr = MLIRHelper::getName(namedTupleMember->name, stringAllocator);

                auto type = getType(namedTupleMember->type);

                assert(type);
                types.push_back({mcl.TupleFieldName(namePtr), type});
            }
            else if (typeItem == SyntaxKind::LiteralType)
            {
                auto literalTypeNode = typeItem.as<LiteralTypeNode>();
                GenContext genContext;
                auto literalValue = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
                auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(literalValue.getDefiningOp());
                attrVal = constantOp.valueAttr();
                continue;
            }
            else
            {
                auto type = getType(typeItem);

                assert(type);
                types.push_back({attrVal, type});
            }

            attrVal = mlir::Attribute();
        }
    }

    void getTupleFieldInfo(TypeLiteralNode typeLiteral, mlir::SmallVector<mlir_ts::FieldInfo> &types)
    {
        for (auto typeItem : typeLiteral->members)
        {
            if (typeItem == SyntaxKind::PropertySignature)
            {
                auto propertySignature = typeItem.as<PropertySignature>();
                auto namePtr = MLIRHelper::getName(propertySignature->name, stringAllocator);

                auto type = getType(propertySignature->type);

                assert(type);
                types.push_back({mlir::FlatSymbolRefAttr::get(builder.getContext(), namePtr), type});
            }
            else
            {
                auto type = getType(typeItem);

                assert(type);
                types.push_back({mlir::Attribute(), type});
            }
        }
    }

    mlir_ts::ConstTupleType getConstTupleType(TupleTypeNode tupleType)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        getTupleFieldInfo(tupleType, types);
        return getConstTupleType(types);
    }

    mlir_ts::ConstTupleType getConstTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::ConstTupleType::get(builder.getContext(), fieldInfos);
    }

    mlir_ts::TupleType getTupleType(TupleTypeNode tupleType)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        getTupleFieldInfo(tupleType, types);
        return getTupleType(types);
    }

    mlir_ts::TupleType getTupleType(TypeLiteralNode typeLiteral)
    {
        mlir::SmallVector<mlir_ts::FieldInfo> types;
        getTupleFieldInfo(typeLiteral, types);
        return getTupleType(types);
    }

    mlir_ts::TupleType getTupleType(mlir::SmallVector<mlir_ts::FieldInfo> &fieldInfos)
    {
        return mlir_ts::TupleType::get(builder.getContext(), fieldInfos);
    }

    mlir::FunctionType getFunctionType(FunctionTypeNode functionType)
    {
        auto resultType = getType(functionType->type);
        SmallVector<mlir::Type> argTypes;
        for (auto paramItem : functionType->parameters)
        {
            argTypes.push_back(getType(paramItem->type));
        }

        return mlir::FunctionType::get(builder.getContext(), argTypes, resultType);
    }

    mlir::Type getUnionType(UnionTypeNode unionTypeNode)
    {
        mlir::SmallVector<mlir::Type> types;
        auto oneType = true;
        mlir::Type currentType;
        for (auto typeItem : unionTypeNode->types)
        {
            auto type = getType(typeItem);
            if (!type)
            {
                llvm_unreachable("wrong type");
            }

            if (currentType && currentType != type)
            {
                oneType = false;
            }

            currentType = type;

            types.push_back(type);
        }

        if (oneType)
        {
            return currentType;
        }

        return getUnionType(types);
    }

    mlir_ts::UnionType getUnionType(mlir::SmallVector<mlir::Type> &types)
    {
        return mlir_ts::UnionType::get(builder.getContext(), types);
    }

    mlir_ts::IntersectionType getIntersectionType(IntersectionTypeNode intersectionTypeNode)
    {
        mlir::SmallVector<mlir::Type> types;
        for (auto typeItem : intersectionTypeNode->types)
        {
            auto type = getType(typeItem);
            if (!type)
            {
                llvm_unreachable("wrong type");
            }

            types.push_back(type);
        }

        return getIntersectionType(types);
    }

    mlir_ts::IntersectionType getIntersectionType(mlir::SmallVector<mlir::Type> &types)
    {
        return mlir_ts::IntersectionType::get(builder.getContext(), types);
    }

    mlir::Type getParenthesizedType(ParenthesizedTypeNode parenthesizedTypeNode)
    {
        return getType(parenthesizedTypeNode->type);
    }

    mlir::Type getLiteralType(LiteralTypeNode literalTypeNode)
    {
        GenContext genContext;
        genContext.dummyRun = true;
        genContext.allowPartialResolve = true;
        auto value = mlirGen(literalTypeNode->literal.as<Expression>(), genContext);
        auto type = value.getType();
        return type;
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

    mlir::LogicalResult declare(VariableDeclarationDOM::TypePtr var, mlir::Value value, bool redeclare = false)
    {
        const auto &name = var->getName();
        /*
        if (!redeclare && symbolTable.count(name))
        {
            return mlir::failure();
        }
        */

        symbolTable.insert(name, {value, var});
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

    auto getNameWithoutNamespace(StringRef name) -> StringRef
    {
        auto pos = name.find_last_of('.');
        if (pos == StringRef::npos)
        {
            return name;
        }

        return name.substr(pos + 1);
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

    auto getGlobalsMap() -> llvm::StringMap<VariableDeclarationDOM::TypePtr> &
    {
        return currentNamespace->globalsMap;
    }

    auto getCaptureVarsMap() -> llvm::StringMap<llvm::StringMap<VariablePairT>> &
    {
        return currentNamespace->captureVarsMap;
    }

    auto getClassesMap() -> llvm::StringMap<ClassInfo::TypePtr> &
    {
        return currentNamespace->classesMap;
    }

    auto getEnumsMap() -> llvm::StringMap<std::pair<mlir::Type, mlir::DictionaryAttr>> &
    {
        return currentNamespace->enumsMap;
    }

    auto getTypeAliasMap() -> llvm::StringMap<mlir::Type> &
    {
        return currentNamespace->typeAliasMap;
    }

    auto getImportEqualsMap() -> llvm::StringMap<mlir::StringRef> &
    {
        return currentNamespace->importEqualsMap;
    }

  protected:
    mlir::StringAttr getStringAttr(std::string text)
    {
        return builder.getStringAttr(text);
    }

    /// Helper conversion for a TypeScript AST location to an MLIR location.
    mlir::Location loc(TextRange loc)
    {
        // return builder.getFileLineColLoc(builder.getIdentifier(fileName), loc->pos, loc->_end);
        auto posLineChar = parser.getLineAndCharacterOfPosition(sourceFile, loc->pos);
        return mlir::FileLineColLoc::get(builder.getContext(), builder.getIdentifier(fileName), posLineChar.line + 1,
                                         posLineChar.character + 1);
    }

    /// A "module" matches a TypeScript source file: containing a list of functions.
    mlir::ModuleOp theModule;

    /// The builder is a helper class to create IR inside a function. The builder
    /// is stateful, in particular it keeps an "insertion point": this is where
    /// the next operations will be introduced.
    mlir::OpBuilder builder;

    mlir::StringRef fileName;

    /// An allocator used for alias names.
    llvm::BumpPtrAllocator stringAllocator;

    llvm::ScopedHashTable<StringRef, VariablePairT> symbolTable;

    NamespaceInfo::TypePtr rootNamespace;

    NamespaceInfo::TypePtr currentNamespace;

    llvm::StringMap<NamespaceInfo::TypePtr> fullNamespacesMap;

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
    return wstos(s.str());
}

mlir::OwningModuleRef mlirGenFromSource(const mlir::MLIRContext &context, const llvm::StringRef &fileName, const llvm::StringRef &source)
{
    Parser parser;
    auto sourceFile =
        parser.parseSourceFile(stows(static_cast<std::string>(fileName)), stows(static_cast<std::string>(source)), ScriptTarget::Latest);
    return MLIRGenImpl(context, fileName).mlirGenSourceFile(sourceFile);
}

} // namespace typescript
