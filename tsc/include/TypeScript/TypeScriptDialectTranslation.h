#ifndef MLIR_TARGET_LLVMIR_DIALECT_TYPESCRIPT_TYPESCRIPTTOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_TYPESCRIPT_TYPESCRIPTTOLLVMIRTRANSLATION_H

namespace mlir
{

class DialectRegistry;
class MLIRContext;

namespace typescript
{
/// Register the TypeScript dialect and the translation from it to the LLVM IR in the
/// given registry;
void registerTypeScriptDialectTranslation(DialectRegistry &registry);

/// Register the TypeScript dialect and the translation from it in the registry
/// associated with the given context.
void registerTypeScriptDialectTranslation(MLIRContext &context);
} // namespace typescript

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_TYPESCRIPT_TYPESCRIPTTOLLVMIRTRANSLATION_H
