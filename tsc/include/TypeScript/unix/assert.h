
class AssertOpLowering : public TsLlvmPattern<mlir_ts::AssertOp>
{
  public:
    using TsLlvmPattern<mlir_ts::AssertOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AssertOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());

        auto loc = op->getLoc();

        auto line = 0;
        auto column = 0;
        auto fileName = StringRef("");
        TypeSwitch<LocationAttr>(loc).Case<FileLineColLoc>([&](FileLineColLoc loc) {
            fileName = loc.getFilename();
            line = loc.getLine();
            column = loc.getColumn();
        });

        // Insert the `_assert` declaration if necessary.
        auto i8PtrTy = th.getI8PtrType();
        auto assertFuncOp = ch.getOrInsertFunction(
            "__assert_fail", th.getFunctionType(th.getVoidType(), {i8PtrTy, i8PtrTy, rewriter.getI32Type(), i8PtrTy}));

        // Split block at `assert` operation.
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        // Generate IR to call `assert`.
        auto *failureBlock = rewriter.createBlock(opBlock->getParent());

        std::stringstream msgWithNUL;
        msgWithNUL << op.msg().str();

        auto opHash = std::hash<std::string>{}(msgWithNUL.str());

        std::stringstream msgVarName;
        msgVarName << "m_" << opHash;

        std::stringstream fileVarName;
        fileVarName << "f_" << hash_value(fileName);

        std::stringstream fileWithNUL;
        fileWithNUL << fileName.str();

        auto msgCst = ch.getOrCreateGlobalString(msgVarName.str(), msgWithNUL.str());

        auto fileCst = ch.getOrCreateGlobalString(fileVarName.str(), fileName.str());

        // auto nullCst = rewriter.create<LLVM::NullOp>(loc, getI8PtrType(context));

        Value lineNumberRes = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(line));
        Value funcName = rewriter.create<LLVM::NullOp>(loc, i8PtrTy);

        rewriter.create<LLVM::CallOp>(loc, assertFuncOp, ValueRange{msgCst, fileCst, lineNumberRes, funcName});
        rewriter.create<LLVM::UnreachableOp>(loc);

        // Generate assertion test.
        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, op.arg(), continuationBlock, failureBlock);

        return success();
    }
};
