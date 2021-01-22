#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/MLIRGen.h"
#include "TypeScript/Passes.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace typescript;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(
    cl::Positional,
    cl::desc("<input TypeScript file>"),
    cl::init("-"),
    cl::value_desc("filename"));

namespace
{
    enum InputType
    {
        TypeScript,
        MLIR
    };
}

static cl::opt<enum InputType> inputType(
    "x",
    cl::init(TypeScript),
    cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(TypeScript, "TypeScript", "load the input file as a TypeScript source.")),
    cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

namespace
{
    enum Action
    {
        None,
        DumpAST,
        DumpMLIR,
        DumpMLIRAffine,
        DumpMLIRLLVM,
        DumpLLVMIR,
        RunJIT
    };
}

static cl::opt<enum Action> emitAction(
    "emit",
    cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine", "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(clEnumValN(RunJIT, "jit", "JIT the code and run it by invoking the main function")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module)
{
    // Handle '.TypeScript' input to the compiler.
    if (inputType != InputType::MLIR &&
        !llvm::StringRef(inputFilename).endswith(".mlir"))
    {
        auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
        if (std::error_code ec = fileOrErr.getError())
        {
            llvm::errs() << "Could not open input file: " << ec.message() << "\n";
            return 0;
        }

        module = mlirGenFromSource(context, fileOrErr.get()->getBuffer());
        return !module ? 1 : 0;
    }

    // Otherwise, the input is '.mlir'.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code EC = fileOrErr.getError())
    {
        llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return -1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module)
    {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }

    return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningModuleRef &module)
{
    if (int error = loadMLIR(context, module))
        return error;

    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Check to see what granularity of MLIR we are compiling to.
    bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
    bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

    if (enableOpt || isLoweringToAffine)
    {
        // Inline all functions into main and then delete them.
        pm.addPass(mlir::createInlinerPass());

        // Now that there is only one function, we can infer the shapes of each of
        // the operations.
        mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
        optPM.addPass(mlir::createCanonicalizerPass());
        //optPM.addPass(mlir::typescript::createShapeInferencePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
    }

    if (isLoweringToAffine)
    {
        mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();

        // Partially lower the TypeScript dialect with a few cleanups afterwards.
        optPM.addPass(mlir::typescript::createLowerToAffinePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());

        // Add optimizations if enabled.
        if (enableOpt)
        {
            optPM.addPass(mlir::createLoopFusionPass());
            optPM.addPass(mlir::createMemRefDataFlowOptPass());
        }
    }

    if (isLoweringToLLVM)
    {
        // Finish lowering the TypeScript IR to the LLVM dialect.
        pm.addPass(mlir::typescript::createLowerToLLVMPass());
    }

    if (mlir::failed(pm.run(*module)))
        return 4;
    return 0;
}

int dumpAST()
{
    if (inputType == InputType::MLIR
        && !llvm::StringRef(inputFilename).endswith(".mlir"))
    {
        llvm::errs() << "Can't dump a TypeScript AST when the input is MLIR\n";
        return 5;
    }

    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return 0;
    }

    dumpFromSource(fileOrErr.get()->getBuffer());

    return 0;
}

int dumpLLVMIR(mlir::ModuleOp module)
{
    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule)
    {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    /// Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get()))
    {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }

    llvm::errs() << *llvmModule << "\n";
    return 0;
}

int runJit(mlir::ModuleOp module)
{
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    auto maybeEngine = mlir::ExecutionEngine::create(
        module, /*llvmModuleBuilder=*/nullptr, optPipeline);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invoke("main");
    if (invocationResult)
    {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "TypeScript compiler\n");

    if (emitAction == Action::DumpAST)
        return dumpAST();

    // If we aren't dumping the AST, then we are compiling with/to MLIR.

    mlir::MLIRContext context;
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::typescript::TypeScriptDialect>();

    mlir::OwningModuleRef module;
    if (int error = loadAndProcessMLIR(context, module))
        return error;

    // If we aren't exporting to non-mlir, then we are done.
    bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
    if (isOutputingMLIR)
    {
        module->dump();
        return 0;
    }

    // Check to see if we are compiling to LLVM IR.
    if (emitAction == Action::DumpLLVMIR)
        return dumpLLVMIR(*module);

    // Otherwise, we must be running the jit.
    if (emitAction == Action::RunJIT)
        return runJit(*module);

    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    return -1;
}
