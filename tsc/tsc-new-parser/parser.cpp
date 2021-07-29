#include "parser.h"
#include "node_factory.h"
#include "node_test.h"
#include "utilities.h"

namespace ts
{
namespace Impl
{
// Implement the parser.as<a>() singleton module.  We do this for perf reasons because creating
// parser instances can actually be expensive enough to impact us on projects with many source
// files.
struct Parser
{
    // Share a single scanner across all calls to parse a source file.  This helps speed things
    // up by avoiding the cost of creating/compiling scanners over and over again.
    ts::Scanner scanner;

    NodeFlags disallowInAndDecoratorContext = NodeFlags::DisallowInContext | NodeFlags::DecoratorContext;

    NodeCreateFunc nodeCreateCallback;

    string fileName;
    NodeFlags sourceFlags;
    string sourceText;
    ScriptTarget languageVersion;
    ScriptKind scriptKind;
    LanguageVariant languageVariant;
    std::vector<DiagnosticWithDetachedLocation> parseDiagnostics;
    std::vector<DiagnosticWithDetachedLocation> jsDocDiagnostics;
    IncrementalParser::SyntaxCursor syntaxCursor;

    SyntaxKind currentToken;
    number nodeCount;
    std::map<string, string> identifiers;
    std::map<string, string> privateIdentifiers;
    number identifierCount;

    ParsingContext parsingContext;

    std::vector<number> notParenthesizedArrow;

    // Flags that dictate what parsing context we're in.  For example:
    // Whether or not we are in strict parsing mode.  All that changes in strict parsing mode is
    // that some tokens that would be considered identifiers may be considered keywords.
    //
    // When adding more parser context flags, consider which is the more common case that the
    // flag will be in.  This should be the 'false' state for that flag.  The reason for this is
    // that we don't store data in our nodes unless the value is in the *non-default* state.  So,
    // for example, more often than code 'allows-in' (or doesn't 'disallow-in').  We opt for
    // 'disallow-in' set to 'false'.  Otherwise, if we had 'allowsIn' set to 'true', then almost
    // all nodes would need extra state on them to store this info.
    //
    // Note: 'allowIn' and 'allowYield' track 1:1 with the [in] and [yield] concepts in the ES6
    // grammar specification.
    //
    // An important thing about these context concepts.  By default they are effectively inherited
    // while parsing through every grammar production.  i.e. if you don't change them, then when
    // you parse a sub-production, it will have the same context values.as<the>() parent production.
    // This is great most of the time.  After all, consider all the 'expression' grammar productions
    // and how nearly all of them pass along the 'in' and 'yield' context values:
    //
    // EqualityExpression[In, Yield] :
    //      RelationalExpression[?In, ?Yield]
    //      EqualityExpression[?In, ?Yield] == RelationalExpression[?In, ?Yield]
    //      EqualityExpression[?In, ?Yield] != RelationalExpression[?In, ?Yield]
    //      EqualityExpression[?In, ?Yield] == RelationalExpression[?In, ?Yield]
    //      EqualityExpression[?In, ?Yield] != RelationalExpression[?In, ?Yield]
    //
    // Where you have to be careful is then understanding what the points are in the grammar
    // where the values are *not* passed along.  For example:
    //
    // SingleNameBinding[Yield,GeneratorParameter]
    //      [+GeneratorParameter]BindingIdentifier[Yield] Initializer[In]opt
    //      [~GeneratorParameter]BindingIdentifier[?Yield]Initializer[In, ?Yield]opt
    //
    // Here this is saying that if the GeneratorParameter context flag is set, that we should
    // explicitly set the 'yield' context flag to false before calling into the BindingIdentifier
    // and we should explicitly unset the 'yield' context flag before calling into the Initializer.
    // production.  Conversely, if the GeneratorParameter context flag is not set, then we
    // should leave the 'yield' context flag alone.
    //
    // Getting this all correct is tricky and requires careful reading of the grammar to
    // understand when these values should be changed versus when they should be inherited.
    //
    // it Note should not be necessary to save/restore these flags during speculative/lookahead
    // parsing.  These context flags are naturally stored and restored through normal recursive
    // descent parsing and unwinding.
    NodeFlags contextFlags;

    // Indicates whether we are currently parsing top-level statements.
    boolean topLevel = true;

    // Whether or not we've had a parse error since creating the last AST node->  If we have
    // encountered an error, it will be stored on the next AST node we create.  Parse errors
    // can be broken down into three categories:
    //
    // 1) An error that occurred during scanning.  For example, an unterminated literal, or a
    //    character that was completely not understood.
    //
    // 2) A token was expected, but was not present.  This type of error is commonly produced
    //    by the 'parseExpected' function.
    //
    // 3) A token was present that no parsing auto was able to consume.  This type of error
    //    only occurs in the 'abortParsingListOrMoveToNextToken' auto when the parser
    //    decides to skip the token.
    //
    // In all of these cases, we want to mark the next node.as<having>() had an error before it.
    // With this mark, we can know in incremental settings if this node can be reused, or if
    // we have to reparse it.  If we don't keep this information around, we may just reuse the
    // node->  in that event we would then not produce the same errors.as<we>() did before, causing
    // significant confusion problems.
    //
    // it Note is necessary that this value be saved/restored during speculative/lookahead
    // parsing.  During lookahead parsing, we will often create a node->  That node will have
    // this value attached, and then this value will be set back to 'false'.  If we decide to
    // rewind, we must get back to the same value we had prior to the lookahead.
    //
    // any Note errors at the end of the file that do not precede a regular node, should get
    // attached to the EOF token.
    boolean parseErrorBeforeNextFinishedNode = false;

    // Rather than using `createBaseNodeFactory` here, we establish a `BaseNodeFactory` that closes over the
    // constructors above, which are reset each time `initializeState` is called.
    NodeFactory factory;

    // Share a single scanner across all calls to parse a source file.  This helps speed things
    // up by avoiding the cost of creating/compiling scanners over and over again.
    Parser()
        : scanner(ScriptTarget::Latest, /*skipTrivia*/ true),
          factory(&scanner, NodeFactoryFlags::NoParenthesizerRules | NodeFactoryFlags::NoNodeConverters | NodeFactoryFlags::NoOriginalNode,
                  [&](Node node) -> void { countNode(node); })
    {
    }

    auto countNode(Node node) -> Node
    {
        nodeCount++;
        return node;
    }

    auto parseSourceFile(string fileName, string sourceText, ScriptTarget languageVersion, IncrementalParser::SyntaxCursor syntaxCursor,
                         boolean setParentNodes = false, ScriptKind scriptKind = ScriptKind::Unknown) -> SourceFile
    {
        scriptKind = ensureScriptKind(fileName, scriptKind);
        if (scriptKind == ScriptKind::JSON)
        {
            auto result = parseJsonText(fileName, sourceText, languageVersion, syntaxCursor, setParentNodes);
            // TODO: review if we need it
            // convertToObjectWorker(result, result.statements[0].expression, result.parseDiagnostics, /*returnValue*/ false,
            // /*knownRootOptions*/ undefined, /*jsonConversionNotifier*/ undefined);
            result->referencedFiles.clear();
            result->typeReferenceDirectives.clear();
            result->libReferenceDirectives.clear();
            result->amdDependencies.clear();
            result->hasNoDefaultLib = false;
            result->pragmas.clear();
            return result;
        }

        initializeState(fileName, sourceText, languageVersion, syntaxCursor, scriptKind);

        auto result = parseSourceFileWorker(languageVersion, setParentNodes, scriptKind);

        clearState();

        return result;
    }

    auto parseIsolatedEntityName(string content, ScriptTarget languageVersion) -> EntityName
    {
        // Choice of `isDeclarationFile` should be arbitrary
        initializeState(string(), content, languageVersion, undefined, ScriptKind::JS);
        // Prime the scanner.
        nextToken();
        auto entityName = parseEntityName(/*allowReservedWords*/ true);
        auto isInvalid = token() == SyntaxKind::EndOfFileToken && !parseDiagnostics.size();
        clearState();
        return isInvalid ? entityName : undefined;
    }

    auto fixupParentReferences(Node rootNode) -> void
    {
        // normally parent references are set during binding. However, for clients that only need
        // a syntax tree, and no semantic features, then the binding process is an unnecessary
        // overhead.  This functions allows us to set all the parents, without all the expense of
        // binding.
        setParentRecursive<boolean>(rootNode, /*incremental*/ true);
    }

    /**
     * Parse json text into SyntaxTree and return node and parse errors if any
     * @param fileName
     * @param sourceText
     */
    auto parseJsonText(string fileName, string sourceText, ScriptTarget languageVersion = ScriptTarget::ES2015,
                       IncrementalParser::SyntaxCursor syntaxCursor = undefined, boolean setParentNodes = false) -> JsonSourceFile
    {
        initializeState(fileName, sourceText, languageVersion, syntaxCursor, ScriptKind::JSON);
        sourceFlags = contextFlags;

        // Prime the scanner.
        nextToken();
        auto pos = getNodePos();
        NodeArray<Statement> statements;
        Node endOfFileToken;
        if (token() == SyntaxKind::EndOfFileToken)
        {
            statements = createNodeArray(NodeArray<Statement>(), pos, pos);
            endOfFileToken = parseTokenNode<EndOfFileToken>();
        }
        else
        {
            // Loop and synthesize an ArrayLiteralExpression if there are more than
            // one top-level expressions to ensure all input text is consumed.
            NodeArray<Expression> expressions;
            while (token() != SyntaxKind::EndOfFileToken)
            {
                Node expression;
                switch (token())
                {
                case SyntaxKind::OpenBracketToken:
                    expression = parseArrayLiteralExpression();
                    break;
                case SyntaxKind::TrueKeyword:
                case SyntaxKind::FalseKeyword:
                    expression = parseTokenNode<BooleanLiteral>();
                    break;
                case SyntaxKind::NullKeyword:
                    expression = parseTokenNode<NullLiteral>();
                    break;
                case SyntaxKind::MinusToken:
                    if (lookAhead<boolean>(
                            [&]() { return nextToken() == SyntaxKind::NumericLiteral && nextToken() != SyntaxKind::ColonToken; }))
                    {
                        expression = parsePrefixUnaryExpression();
                    }
                    else
                    {
                        expression = parseObjectLiteralExpression();
                    }
                    break;
                case SyntaxKind::NumericLiteral:
                case SyntaxKind::StringLiteral:
                    if (lookAhead<boolean>([&]() { return nextToken() != SyntaxKind::ColonToken; }))
                    {
                        expression = parseLiteralNode();
                        break;
                    }
                    // falls through
                default:
                    expression = parseObjectLiteralExpression();
                    break;
                }

                // Error collect recovery multiple top-level expressions
                expressions.push_back(expression);
                if (token() != SyntaxKind::EndOfFileToken)
                {
                    parseErrorAtCurrentToken(ts::DiagnosticMessage(data::DiagnosticMessage(Diagnostics::Unexpected_token)));
                }
            }

            auto expression = expressions.size() > 1 ? finishNode(factory.createArrayLiteralExpression(expressions), pos).as<Expression>()
                                                     : Debug::checkDefined(expressions[0]);
            auto statement = factory.createExpressionStatement(expression).as<Statement>();
            finishNode(statement, pos);
            statements = createNodeArray(NodeArray<Statement>(statement), pos);
            endOfFileToken = parseExpectedToken(SyntaxKind::EndOfFileToken, data::DiagnosticMessage(Diagnostics::Unexpected_token));
        }

        // Set source file so that errors will be reported with this file name
        auto sourceFile = createSourceFile(fileName, ScriptTarget::ES2015, ScriptKind::JSON, /*isDeclaration*/ false, statements,
                                           endOfFileToken, sourceFlags);

        if (setParentNodes)
        {
            fixupParentReferences(sourceFile);
        }

        sourceFile->nodeCount = nodeCount;
        sourceFile->identifierCount = identifierCount;
        sourceFile->identifiers = identifiers;
        // sourceFile->parseDiagnostics = attachFileToDiagnostics(parseDiagnostics, sourceFile);
        copy(sourceFile->parseDiagnostics, attachFileToDiagnostics(parseDiagnostics, sourceFile));
        if (!jsDocDiagnostics.empty())
        {
            // sourceFile->jsDocDiagnostics = attachFileToDiagnostics(jsDocDiagnostics, sourceFile);
            copy(sourceFile->jsDocDiagnostics, attachFileToDiagnostics(jsDocDiagnostics, sourceFile));
        }

        auto result = JsonSourceFile(sourceFile);
        clearState();
        return result;
    }

    auto initializeState(string _fileName, string _sourceText, ScriptTarget _languageVersion, IncrementalParser::SyntaxCursor _syntaxCursor,
                         ScriptKind _scriptKind) -> void
    {
        fileName = normalizePath(_fileName);
        sourceText = _sourceText;
        languageVersion = _languageVersion;
        syntaxCursor = _syntaxCursor;
        scriptKind = _scriptKind;
        languageVariant = getLanguageVariant(_scriptKind);

        parseDiagnostics.clear();
        parsingContext = ParsingContext::Unknown;
        identifiers.clear();
        privateIdentifiers.clear();
        identifierCount = 0;
        nodeCount = 0;
        sourceFlags = NodeFlags::None;
        topLevel = true;

        switch (scriptKind)
        {
        case ScriptKind::JS:
        case ScriptKind::JSX:
            contextFlags = NodeFlags::JavaScriptFile;
            break;
        case ScriptKind::JSON:
            contextFlags = NodeFlags::JavaScriptFile | NodeFlags::JsonFile;
            break;
        default:
            contextFlags = NodeFlags::None;
            break;
        }
        parseErrorBeforeNextFinishedNode = false;

        // Initialize and prime the scanner before parsing the source elements.
        scanner.setText(sourceText);
        scanner.setOnError(std::bind(&Parser::scanError, this, std::placeholders::_1, std::placeholders::_2));
        scanner.setScriptTarget(languageVersion);
        scanner.setLanguageVariant(languageVariant);
    }

    auto clearState() -> void
    {
        // Clear out the text the scanner is pointing at, so it doesn't keep anything alive unnecessarily.
        scanner.clearCommentDirectives();
        scanner.setText(string());
        scanner.setOnError(nullptr);

        // Clear any data.  We don't want to accidentally hold onto it for too long.
        sourceText = string();
        languageVersion = ScriptTarget::ES3;
        syntaxCursor = undefined;
        scriptKind = ScriptKind::Unknown;
        languageVariant = LanguageVariant::Standard;
        sourceFlags = NodeFlags::None;
        parseDiagnostics.clear();
        jsDocDiagnostics.clear();
        parsingContext = ParsingContext::Unknown;
        identifiers.clear();
        notParenthesizedArrow.clear();
        topLevel = true;
    }

    /** @internal */
    auto isDeclarationFileName(string fileName) -> boolean
    {
        return fileExtensionIs(fileName, Extension::Dts);
    }

    auto parseSourceFileWorker(ScriptTarget languageVersion, boolean setParentNodes, ScriptKind scriptKind) -> SourceFile
    {
        auto isDeclarationFile = isDeclarationFileName(fileName);
        if (isDeclarationFile)
        {
            contextFlags |= NodeFlags::Ambient;
        }

        sourceFlags = contextFlags;

        // Prime the scanner.
        nextToken();

        auto statements = parseList<Statement>(ParsingContext::SourceElements, std::bind(&Parser::parseStatement, this));
        Debug::_assert(token() == SyntaxKind::EndOfFileToken);
        auto endOfFileToken = addJSDocComment(parseTokenNode<EndOfFileToken>());

        auto sourceFile =
            createSourceFile(fileName, languageVersion, scriptKind, isDeclarationFile, statements, endOfFileToken, sourceFlags);

        // A member of ReadonlyArray<T> isn't assignable to a member of T[] (and prevents a direct cast) - but this is where we set up those
        // members so they can be in the future
        processCommentPragmas(sourceFile, sourceText);

        auto reportPragmaDiagnostic = [&](number pos, number end, DiagnosticMessage diagnostic) -> void {
            parseDiagnostics.push_back(createDetachedDiagnostic(fileName, pos, end, diagnostic));
        };
        processPragmasIntoFields(sourceFile, reportPragmaDiagnostic);

        copy(sourceFile->commentDirectives, scanner.getCommentDirectives());
        sourceFile->nodeCount = nodeCount;
        sourceFile->identifierCount = identifierCount;
        sourceFile->identifiers = identifiers;
        // sourceFile->parseDiagnostics = attachFileToDiagnostics(parseDiagnostics, sourceFile);
        copy(sourceFile->parseDiagnostics, attachFileToDiagnostics(parseDiagnostics, sourceFile));
        if (!jsDocDiagnostics.empty())
        {
            // sourceFile->jsDocDiagnostics = attachFileToDiagnostics(jsDocDiagnostics, sourceFile);
            copy(sourceFile->jsDocDiagnostics, attachFileToDiagnostics(jsDocDiagnostics, sourceFile));
        }

        if (setParentNodes)
        {
            fixupParentReferences(sourceFile);
        }

        return sourceFile;
    }

    template <typename T> auto withJSDoc(T node, boolean hasJSDoc) -> T
    {
        return hasJSDoc ? addJSDocComment(node) : node;
    }

    boolean hasDeprecatedTag = false;
    template <typename T> auto addJSDocComment(T node) -> T
    {
        // TODO:
        // Debug::_assert(!node.as<JSDocContainer>()->jsDoc); // Should only be called once per node
        /*
        auto jsDoc = mapDefined(getJSDocCommentRanges(node, sourceText), [&] (auto comment) { return JSDocParser::parseJSDocComment(node,
        comment->pos, comment->_end - comment->pos); }); if (jsDoc.size()) node->jsDoc = jsDoc; if (hasDeprecatedTag) { hasDeprecatedTag =
        false; node->flags |= NodeFlags::Deprecated;
        }
        */
        return node;
    }

    auto reparseTopLevelAwait(SourceFile sourceFile) -> SourceFile
    {
        auto savedSyntaxCursor = syntaxCursor;
        auto baseSyntaxCursor = IncrementalParser::createSyntaxCursor(sourceFile);

        auto containsPossibleTopLevelAwait = [](Node node) {
            return !(node->flags & NodeFlags::AwaitContext) && !!(node->transformFlags & TransformFlags::ContainsPossibleTopLevelAwait);
        };

        auto findNextStatementWithAwait = [&](NodeArray<Statement> statements, number start) {
            for (auto i = start; i < statements.size(); i++)
            {
                if (containsPossibleTopLevelAwait(statements[i]))
                {
                    return i;
                }
            }
            return -1;
        };

        auto findNextStatementWithoutAwait = [&](NodeArray<Statement> statements, number start) {
            for (auto i = start; i < statements.size(); i++)
            {
                if (!containsPossibleTopLevelAwait(statements[i]))
                {
                    return i;
                }
            }
            return -1;
        };

        auto currentNode = [&](number position) {
            auto node = baseSyntaxCursor.currentNode(position);
            if (topLevel && node && containsPossibleTopLevelAwait(node))
            {
                node.intersectsChange = true;
            }
            return node;
        };

        syntaxCursor = IncrementalParser::SyntaxCursor{currentNode};

        NodeArray<Statement> statements;
        auto savedParseDiagnostics = parseDiagnostics;

        parseDiagnostics.clear();

        auto pos = 0;
        auto start = findNextStatementWithAwait(sourceFile->statements, 0);
        while (start != -1)
        {
            // append all statements between pos and start
            auto prevStatement = sourceFile->statements[pos];
            auto nextStatement = sourceFile->statements[start];
            addRange(statements, sourceFile->statements, pos, start);
            pos = findNextStatementWithoutAwait(sourceFile->statements, start);

            // append all diagnostics associated with the copied range
            auto diagnosticStart =
                findIndex(savedParseDiagnostics, [&](auto diagnostic, number index) { return diagnostic->start >= prevStatement->pos; });
            auto diagnosticEnd =
                diagnosticStart >= 0
                    ? findIndex(
                          savedParseDiagnostics, [&](auto diagnostic, number index) { return diagnostic->start >= nextStatement->pos; },
                          diagnosticStart)
                    : -1;
            if (diagnosticStart >= 0)
            {
                addRange(parseDiagnostics, savedParseDiagnostics, diagnosticStart, diagnosticEnd >= 0 ? diagnosticEnd : -1);
            }

            // reparse all statements between start and pos. We skip existing diagnostics for the same range and allow the parser to
            // generate new ones.
            speculationHelper<boolean>(
                [&]() {
                    auto savedContextFlags = contextFlags;
                    contextFlags |= NodeFlags::AwaitContext;
                    scanner.setTextPos(nextStatement->pos);
                    nextToken();

                    while (token() != SyntaxKind::EndOfFileToken)
                    {
                        auto startPos = scanner.getStartPos();
                        auto statement =
                            parseListElement<Statement>(ParsingContext::SourceElements, std::bind(&Parser::parseStatement, this));
                        statements.push_back(statement);
                        if (startPos == scanner.getStartPos())
                        {
                            nextToken();
                        }

                        if (pos >= 0)
                        {
                            auto nonAwaitStatement = sourceFile->statements[pos];
                            if (statement->_end == nonAwaitStatement->pos)
                            {
                                // done reparsing this section
                                break;
                            }
                            if (statement->_end > nonAwaitStatement->pos)
                            {
                                // we ate into the next statement, so we must reparse it.
                                pos = findNextStatementWithoutAwait(sourceFile->statements, pos + 1);
                            }
                        }
                    }

                    contextFlags = savedContextFlags;

                    return true;
                },
                SpeculationKind::Reparse);

            // find the next statement containing an `await`
            start = pos >= 0 ? findNextStatementWithAwait(sourceFile->statements, pos) : -1;
        }

        // append all statements between pos and the end of the list
        if (pos >= 0)
        {
            auto prevStatement = sourceFile->statements[pos];
            addRange(statements, sourceFile->statements, pos);

            // append all diagnostics associated with the copied range
            auto diagnosticStart =
                findIndex(savedParseDiagnostics, [&](auto diagnostic, number index) { return diagnostic->start >= prevStatement->pos; });
            if (diagnosticStart >= 0)
            {
                addRange(parseDiagnostics, savedParseDiagnostics, diagnosticStart);
            }
        }

        syntaxCursor = savedSyntaxCursor;
        return factory.updateSourceFile(sourceFile,
                                        setTextRange(factory.createNodeArray(statements), (data::TextRange)sourceFile->statements));
    }

    auto createSourceFile(string fileName, ScriptTarget languageVersion, ScriptKind scriptKind, boolean isDeclarationFile,
                          NodeArray<Statement> statements, Node endOfFileToken, NodeFlags flags) -> SourceFile
    {
        // code from createNode is inlined here so createNode won't have to deal with special case of creating source files
        // this is quite rare comparing to other nodes and createNode should be.as<fast>().as<possible>()
        auto sourceFile = factory.createSourceFile(statements, endOfFileToken, flags);
        setTextRangePosWidth(sourceFile, 0, sourceText.size());
        setExternalModuleIndicator(sourceFile);

        // If we parsed this.as<an>() external module, it may contain top-level await
        if (!isDeclarationFile && isExternalModule(sourceFile) &&
            !!(sourceFile->transformFlags & TransformFlags::ContainsPossibleTopLevelAwait))
        {
            sourceFile = reparseTopLevelAwait(sourceFile);
        }

        sourceFile->text = sourceText;
        sourceFile->bindDiagnostics.clear();
        sourceFile->bindSuggestionDiagnostics.clear();
        sourceFile->languageVersion = languageVersion;
        sourceFile->fileName = fileName;
        sourceFile->languageVariant = getLanguageVariant(scriptKind);
        sourceFile->isDeclarationFile = isDeclarationFile;
        sourceFile->scriptKind = scriptKind;

        return sourceFile;
    }

    auto setContextFlag(boolean val, NodeFlags flag)
    {
        if (val)
        {
            contextFlags |= flag;
        }
        else
        {
            contextFlags &= ~flag;
        }
    }

    auto setDisallowInContext(boolean val)
    {
        setContextFlag(val, NodeFlags::DisallowInContext);
    }

    auto setYieldContext(boolean val)
    {
        setContextFlag(val, NodeFlags::YieldContext);
    }

    auto setDecoratorContext(boolean val)
    {
        setContextFlag(val, NodeFlags::DecoratorContext);
    }

    auto setAwaitContext(boolean val)
    {
        setContextFlag(val, NodeFlags::AwaitContext);
    }

    template <typename T> auto doOutsideOfContext(NodeFlags context, std::function<T()> func) -> T
    {
        // contextFlagsToClear will contain only the context flags that are
        // currently set that we need to temporarily clear
        // We don't just blindly reset to the previous flags to ensure
        // that we do not mutate cached flags for the incremental
        // parser (ThisNodeHasError, ThisNodeOrAnySubNodesHasError, and
        // HasAggregatedChildData).
        auto contextFlagsToClear = context & contextFlags;
        if (!!contextFlagsToClear)
        {
            // clear the requested context flags
            setContextFlag(/*val*/ false, contextFlagsToClear);
            auto result = func();
            // restore the context flags we just cleared
            setContextFlag(/*val*/ true, contextFlagsToClear);
            return result;
        }

        // no need to do anything special.as<we>() are not in any of the requested contexts
        return func();
    }

    template <typename T> auto doInsideOfContext(NodeFlags context, std::function<T()> func) -> T
    {
        // contextFlagsToSet will contain only the context flags that
        // are not currently set that we need to temporarily enable.
        // We don't just blindly reset to the previous flags to ensure
        // that we do not mutate cached flags for the incremental
        // parser (ThisNodeHasError, ThisNodeOrAnySubNodesHasError, and
        // HasAggregatedChildData).
        auto contextFlagsToSet = context & ~contextFlags;
        if (!!contextFlagsToSet)
        {
            // set the requested context flags
            setContextFlag(/*val*/ true, contextFlagsToSet);
            auto result = func();
            // reset the context flags we just set
            setContextFlag(/*val*/ false, contextFlagsToSet);
            return result;
        }

        // no need to do anything special.as<we>() are already in all of the requested contexts
        return func();
    }

    template <typename T> auto allowInAnd(std::function<T()> func) -> T
    {
        return doOutsideOfContext(NodeFlags::DisallowInContext, func);
    }

    template <typename T> auto disallowInAnd(std::function<T()> func) -> T
    {
        return doInsideOfContext(NodeFlags::DisallowInContext, func);
    }

    template <typename T> auto doInYieldContext(std::function<T()> func) -> T
    {
        return doInsideOfContext(NodeFlags::YieldContext, func);
    }

    template <typename T> auto doInDecoratorContext(std::function<T()> func) -> T
    {
        return doInsideOfContext(NodeFlags::DecoratorContext, func);
    }

    template <typename T> auto doInAwaitContext(std::function<T()> func) -> T
    {
        return doInsideOfContext(NodeFlags::AwaitContext, func);
    }

    template <typename T> auto doOutsideOfAwaitContext(std::function<T()> func) -> T
    {
        return doOutsideOfContext(NodeFlags::AwaitContext, func);
    }

    template <typename T> auto doInYieldAndAwaitContext(std::function<T()> func) -> T
    {
        return doInsideOfContext(NodeFlags::YieldContext | NodeFlags::AwaitContext, func);
    }

    template <typename T> auto doOutsideOfYieldAndAwaitContext(std::function<T()> func) -> T
    {
        return doOutsideOfContext(NodeFlags::YieldContext | NodeFlags::AwaitContext, func);
    }

    auto inContext(NodeFlags flags)
    {
        return (contextFlags & flags) != NodeFlags::None;
    }

    auto inYieldContext()
    {
        return inContext(NodeFlags::YieldContext);
    }

    auto inDisallowInContext()
    {
        return inContext(NodeFlags::DisallowInContext);
    }

    auto inDecoratorContext()
    {
        return inContext(NodeFlags::DecoratorContext);
    }

    auto inAwaitContext()
    {
        return inContext(NodeFlags::AwaitContext);
    }

    auto parseErrorAtCurrentToken(DiagnosticMessage message, string arg0 = string()) -> void
    {
        parseErrorAt(scanner.getTokenPos(), scanner.getTextPos(), message, arg0);
    }

    auto parseErrorAtPosition(number start, number length, DiagnosticMessage message, string arg0 = string()) -> void
    {
        // Don't report another error if it would just be at the same position.as<the>() last error.
        auto lastError = lastOrUndefined(parseDiagnostics);
        if (!lastError || start != lastError->start)
        {
            parseDiagnostics.push_back(createDetachedDiagnostic(fileName, start, length, message, arg0));
        }

        // Mark that we've encountered an error.  We'll set an appropriate bit on the next
        // node we finish so that it can't be reused incrementally.
        parseErrorBeforeNextFinishedNode = true;
    }

    auto parseErrorAt(number start, number end, DiagnosticMessage message) -> void
    {
        parseErrorAtPosition(start, end - start, message);
    }

    auto parseErrorAtRange(TextRange range, DiagnosticMessage message) -> void
    {
        parseErrorAt(range->pos, range->_end, message);
    }

    template <typename T> auto parseErrorAt(number start, number end, DiagnosticMessage message, T arg0) -> void
    {
        parseErrorAtPosition(start, end - start, message, arg0);
    }

    template <typename T> auto parseErrorAtRange(TextRange range, DiagnosticMessage message, T arg0) -> void
    {
        parseErrorAt(range->pos, range->_end, message, arg0);
    }

    auto scanError(DiagnosticMessage message, number length) -> void
    {
        parseErrorAtPosition(scanner.getTextPos(), length, message);
    }

    auto getNodePos() -> pos_type
    {
        return {scanner.getStartPos(), scanner.getTokenPos()};
    }

    auto hasPrecedingJSDocComment()
    {
        return scanner.hasPrecedingJSDocComment();
    }

    // Use this auto to access the current token instead of reading the currentToken
    // variable. Since auto results aren't narrowed in control flow analysis, this ensures
    // that the type checker doesn't make wrong assumptions about the type of the current
    // token (e.g. a call to nextToken() changes the current token but the checker doesn't
    // reason about this side effect).  Mainstream VMs inline simple functions like this, so
    // there is no performance penalty.
    auto token() -> SyntaxKind
    {
        return currentToken;
    }

    auto nextTokenWithoutCheck()
    {
        return currentToken = scanner.scan();
    }

    template <typename T> auto nextTokenAnd(std::function<T()> func) -> T
    {
        nextToken();
        return func();
    }

    auto nextToken() -> SyntaxKind
    {
        // if the keyword had an escape
        if (isKeyword(currentToken) && (scanner.hasUnicodeEscape() || scanner.hasExtendedUnicodeEscape()))
        {
            // issue a parse error for the escape
            parseErrorAt(scanner.getTokenPos(), scanner.getTextPos(),
                         data::DiagnosticMessage(Diagnostics::Keywords_cannot_contain_escape_characters));
        }
        return nextTokenWithoutCheck();
    }

    auto nextTokenJSDoc() -> SyntaxKind
    {
        return currentToken = scanner.scanJsDocToken();
    }

    auto reScanGreaterToken() -> SyntaxKind
    {
        return currentToken = scanner.reScanGreaterToken();
    }

    auto reScanSlashToken() -> SyntaxKind
    {
        return currentToken = scanner.reScanSlashToken();
    }

    auto reScanTemplateToken(boolean isTaggedTemplate) -> SyntaxKind
    {
        return currentToken = scanner.reScanTemplateToken(isTaggedTemplate);
    }

    auto reScanTemplateHeadOrNoSubstitutionTemplate() -> SyntaxKind
    {
        return currentToken = scanner.reScanTemplateHeadOrNoSubstitutionTemplate();
    }

    auto reScanLessThanToken() -> SyntaxKind
    {
        return currentToken = scanner.reScanLessThanToken();
    }

    auto scanJsxIdentifier() -> SyntaxKind
    {
        return currentToken = scanner.scanJsxIdentifier();
    }

    auto scanJsxText() -> SyntaxKind
    {
        return currentToken = scanner.scanJsxToken();
    }

    auto scanJsxAttributeValue() -> SyntaxKind
    {
        return currentToken = scanner.scanJsxAttributeValue();
    }

    template <typename T> auto speculationHelper(std::function<T()> callback, SpeculationKind speculationKind) -> T
    {
        // Keep track of the state we'll need to rollback to if lookahead fails (or if the
        // caller asked us to always reset our state).
        auto saveToken = currentToken;
        // TODO: do we need it?
        // auto saveParseDiagnosticsLength = parseDiagnostics.size();
        auto saveParseErrorBeforeNextFinishedNode = parseErrorBeforeNextFinishedNode;

        // it Note is not actually necessary to save/restore the context flags here.  That's
        // because the saving/restoring of these flags happens naturally through the recursive
        // descent nature of our Parser::  However, we still store this here just so we can
        // assert that invariant holds.
        auto saveContextFlags = contextFlags;

        // If we're only looking ahead, then tell the scanner to only lookahead.as<well>().
        // Otherwise, if we're actually speculatively parsing, then tell the scanner to do the
        // same.
        auto result = speculationKind != SpeculationKind::TryParse ? scanner.lookAhead<T>(callback) : scanner.tryScan<T>(callback);

        Debug::_assert(saveContextFlags == contextFlags);

        // If our callback returned something 'falsy' or we're just looking ahead,
        // then unconditionally restore us to where we were.
        if (!result || speculationKind != SpeculationKind::TryParse)
        {
            currentToken = saveToken;
            if (speculationKind != SpeculationKind::Reparse)
            {
                // TODO: do we need it?
                // parseDiagnostics.size() = saveParseDiagnosticsLength;
            }
            parseErrorBeforeNextFinishedNode = saveParseErrorBeforeNextFinishedNode;
        }

        return result;
    }

    /** Invokes the provided callback then unconditionally restores the parser to the state it
     * was in immediately prior to invoking the callback.  The result of invoking the callback
     * is returned from this function.
     */
    template <typename T> auto lookAhead(std::function<T()> callback) -> T
    {
        return speculationHelper<T>(callback, SpeculationKind::Lookahead);
    }

    /** Invokes the provided callback.  If the callback returns something falsy, then it restores
     * the parser to the state it was in immediately prior to invoking the callback.  If the
     * callback returns something truthy, then the parser state is not rolled back.  The result
     * of invoking the callback is returned from this function.
     */
    template <typename T> auto tryParse(std::function<T()> callback) -> T
    {
        return speculationHelper<T>(callback, SpeculationKind::TryParse);
    }

    auto isBindingIdentifier() -> boolean
    {
        if (token() == SyntaxKind::Identifier)
        {
            return true;
        }
        return token() > SyntaxKind::LastReservedWord;
    }

    // Ignore strict mode flag because we will report an error in type checker instead.
    auto isIdentifier() -> boolean
    {
        if (token() == SyntaxKind::Identifier)
        {
            return true;
        }

        // If we have a 'yield' keyword, and we're in the [yield] context, then 'yield' is
        // considered a keyword and is not an identifier.
        if (token() == SyntaxKind::YieldKeyword && inYieldContext())
        {
            return false;
        }

        // If we have a 'await' keyword, and we're in the [Await] context, then 'await' is
        // considered a keyword and is not an identifier.
        if (token() == SyntaxKind::AwaitKeyword && inAwaitContext())
        {
            return false;
        }

        return token() > SyntaxKind::LastReservedWord;
    }

    auto parseExpected(SyntaxKind kind, DiagnosticMessage diagnosticMessage = undefined, boolean shouldAdvance = true) -> boolean
    {
        if (token() == kind)
        {
            if (shouldAdvance)
            {
                nextToken();
            }
            return true;
        }

        // Report specific message if provided with one.  Otherwise, report generic fallback message.
        if (!!diagnosticMessage)
        {
            parseErrorAtCurrentToken(diagnosticMessage);
        }
        else
        {
            parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::_0_expected), scanner.tokenToString(kind));
        }
        return false;
    }

    auto parseExpectedJSDoc(SyntaxKind kind)
    {
        if (token() == kind)
        {
            nextTokenJSDoc();
            return true;
        }
        parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::_0_expected), scanner.tokenToString(kind));
        return false;
    }

    auto parseOptional(SyntaxKind t) -> boolean
    {
        if (token() == t)
        {
            nextToken();
            return true;
        }
        return false;
    }

    auto parseOptionalToken(SyntaxKind t) -> Node
    {
        if (token() == t)
        {
            return parseTokenNode<Node>();
        }
        return undefined;
    }

    auto parseOptionalTokenJSDoc(SyntaxKind t) -> Node
    {
        if (token() == t)
        {
            return parseTokenNodeJSDoc();
        }
        return undefined;
    }

    auto parseExpectedToken(SyntaxKind t, DiagnosticMessage diagnosticMessage = undefined, string arg0 = string()) -> Node
    {
        return parseOptionalToken(t) || [&]() { return createMissingNode(t, /*reportAtCurrentPosition*/ false, diagnosticMessage, arg0); };
    }

    auto parseExpectedTokenJSDoc(SyntaxKind t) -> Node
    {
        return parseOptionalTokenJSDoc(t) || [&]() {
            return createMissingNode(t, /*reportAtCurrentPosition*/ false, data::DiagnosticMessage(Diagnostics::_0_expected),
                                     scanner.tokenToString(t));
        };
    }

    template <typename T> auto parseTokenNode() -> Node
    {
        auto pos = getNodePos();
        auto kind = token();
        nextToken();
        return finishNode(factory.createToken<T>(kind), pos);
    }

    auto parseTokenNodeJSDoc() -> Node
    {
        auto pos = getNodePos();
        auto kind = token();
        nextTokenJSDoc();
        return finishNode(factory.createToken<Node>(kind), pos);
    }

    auto canParseSemicolon()
    {
        // If there's a real semicolon, then we can always parse it out.
        if (token() == SyntaxKind::SemicolonToken)
        {
            return true;
        }

        // We can parse out an optional semicolon in ASI cases in the following cases.
        return token() == SyntaxKind::CloseBraceToken || token() == SyntaxKind::EndOfFileToken || scanner.hasPrecedingLineBreak();
    }

    auto parseSemicolon() -> boolean
    {
        if (canParseSemicolon())
        {
            if (token() == SyntaxKind::SemicolonToken)
            {
                // consume the semicolon if it was explicitly provided.
                nextToken();
            }

            return true;
        }
        else
        {
            return parseExpected(SyntaxKind::SemicolonToken);
        }
    }

    auto createNodeArray(NodeArray<Node> elements, pos_type pos, number end = -1, boolean hasTrailingComma = false) -> NodeArray<Node>
    {
        auto array = factory.createNodeArray(elements, hasTrailingComma);
        setTextRangePosEnd(array, pos, end != -1 ? end : scanner.getStartPos());
        array->pos.textPos = pos.textPos;
        return array;
    }

    template <typename T>
    auto createNodeArray(NodeArray<T> elements, pos_type pos, number end = -1, boolean hasTrailingComma = false) -> NodeArray<T>
    {
        auto array = factory.createNodeArray<T>(elements, hasTrailingComma);
        setTextRangePosEnd(array, pos, end != -1 ? end : scanner.getStartPos());
        array->pos.textPos = pos.textPos;
        return array;
    }

    // TODO: use template instead of Node method to avoid casts
    auto finishNode(Node node, pos_type pos, number end = -1) -> Node
    {
        setTextRangePosEnd(node, pos, end != -1 ? end : scanner.getStartPos());
        node->pos.textPos = pos.textPos;
        if (!!contextFlags)
        {
            node->flags |= contextFlags;
        }

        // Keep track on the node if we encountered an error while parsing it.  If we did, then
        // we cannot reuse the node incrementally.  Once we've marked this node, clear out the
        // flag so that we don't mark any subsequent nodes.
        if (parseErrorBeforeNextFinishedNode)
        {
            parseErrorBeforeNextFinishedNode = false;
            node->flags |= NodeFlags::ThisNodeHasError;
        }

        return node;
    }

    template <typename T = Node>
    auto createMissingNode(SyntaxKind kind, boolean reportAtCurrentPosition, DiagnosticMessage diagnosticMessage = undefined,
                           string arg0 = string()) -> Node
    {
        if (reportAtCurrentPosition)
        {
            parseErrorAtPosition(scanner.getStartPos(), 0, diagnosticMessage, arg0);
        }
        else if (!!diagnosticMessage)
        {
            parseErrorAtCurrentToken(diagnosticMessage, arg0);
        }

        auto pos = getNodePos();
        auto result = kind == SyntaxKind::Identifier ? factory.createIdentifier(string()).as<Node>()
                      : isTemplateLiteralKind(kind)
                          ? factory.createTemplateLiteralLikeNode(kind, string(), string(), /*templateFlags*/ TokenFlags::None).as<Node>()
                      : kind == SyntaxKind::NumericLiteral
                          ? factory.createNumericLiteral(string(), /*numericLiteralFlags*/ TokenFlags::None).as<Node>()
                      : kind == SyntaxKind::StringLiteral      ? factory.createStringLiteral(string(), /*isSingleQuote*/ false).as<Node>()
                      : kind == SyntaxKind::MissingDeclaration ? factory.createMissingDeclaration().as<Node>()
                                                               : factory.createToken<T>(kind).template as<Node>();
        return finishNode(result, pos);
    }

    auto internIdentifier(string text) -> string
    {
        identifiers[text] = text;
        return identifiers.at(text);
    }

    // An identifier that starts with two underscores has an extra underscore character prepended to it to avoid issues
    // with magic property names like '__proto__'. The 'identifiers' object is used to share a single string instance for
    // each identifier in order to reduce memory consumption.
    auto createIdentifier(boolean isIdentifier, DiagnosticMessage diagnosticMessage = undefined,
                          DiagnosticMessage privateIdentifierDiagnosticMessage = undefined) -> Identifier
    {
        if (isIdentifier)
        {
            identifierCount++;
            auto pos = getNodePos();
            // Store original token kind if it is not just an Identifier so we can report appropriate error later in type checker
            auto originalKeywordKind = token();
            auto text = internIdentifier(scanner.getTokenValue());
            nextTokenWithoutCheck();
            return finishNode(factory.createIdentifier(text, /*typeArguments*/ undefined, originalKeywordKind), pos);
        }

        if (token() == SyntaxKind::PrivateIdentifier)
        {
            parseErrorAtCurrentToken(!!privateIdentifierDiagnosticMessage
                                         ? privateIdentifierDiagnosticMessage
                                         : ts::DiagnosticMessage(data::DiagnosticMessage(
                                               Diagnostics::Private_identifiers_are_not_allowed_outside_class_bodies)));
            return createIdentifier(/*isIdentifier*/ true);
        }

        if (token() == SyntaxKind::Unknown &&
            scanner.tryScan<boolean>([&]() { return scanner.reScanInvalidIdentifier() == SyntaxKind::Identifier; }))
        {
            // Scanner has already recorded an 'Invalid character' error, so no need to add another from the Parser::
            return createIdentifier(/*isIdentifier*/ true);
        }

        identifierCount++;
        // Only for end of file because the error gets reported incorrectly on embedded script tags.
        auto reportAtCurrentPosition = token() == SyntaxKind::EndOfFileToken;

        auto isReservedWord = scanner.isReservedWord();
        auto msgArg = scanner.getTokenText();

        auto defaultMessage = isReservedWord
                                  ? data::DiagnosticMessage(Diagnostics::Identifier_expected_0_is_a_reserved_word_that_cannot_be_used_here)
                                  : data::DiagnosticMessage(Diagnostics::Identifier_expected);

        return createMissingNode<Identifier>(SyntaxKind::Identifier, reportAtCurrentPosition,
                                             !!diagnosticMessage ? diagnosticMessage : ts::DiagnosticMessage(defaultMessage), msgArg);
    }

    auto parseBindingIdentifier(DiagnosticMessage privateIdentifierDiagnosticMessage = undefined)
    {
        return createIdentifier(isBindingIdentifier(), /*diagnosticMessage*/ undefined, privateIdentifierDiagnosticMessage);
    }

    auto parseIdentifier(DiagnosticMessage diagnosticMessage = undefined, DiagnosticMessage privateIdentifierDiagnosticMessage = undefined)
        -> Identifier
    {
        return createIdentifier(isIdentifier(), diagnosticMessage, privateIdentifierDiagnosticMessage);
    }

    auto parseIdentifierName(DiagnosticMessage diagnosticMessage = undefined) -> Identifier
    {
        return createIdentifier(scanner.tokenIsIdentifierOrKeyword(token()), diagnosticMessage);
    }

    auto isLiteralPropertyName() -> boolean
    {
        return scanner.tokenIsIdentifierOrKeyword(token()) || token() == SyntaxKind::StringLiteral || token() == SyntaxKind::NumericLiteral;
    }

    auto parsePropertyNameWorker(boolean allowComputedPropertyNames) -> Node
    {
        if (token() == SyntaxKind::StringLiteral || token() == SyntaxKind::NumericLiteral)
        {
            auto node = parseLiteralNode();
            node->text = internIdentifier(node->text);
            return node;
        }
        if (allowComputedPropertyNames && token() == SyntaxKind::OpenBracketToken)
        {
            return parseComputedPropertyName();
        }
        if (token() == SyntaxKind::PrivateIdentifier)
        {
            return parsePrivateIdentifier();
        }
        return parseIdentifierName();
    }

    auto parsePropertyName() -> Node
    {
        return parsePropertyNameWorker(/*allowComputedPropertyNames*/ true);
    }

    auto parseComputedPropertyName() -> Node
    {
        // PropertyName [Yield]:
        //      LiteralPropertyName
        //      ComputedPropertyName[?Yield]
        auto pos = getNodePos();
        parseExpected(SyntaxKind::OpenBracketToken);
        // We parse any expression (including a comma expression). But the grammar
        // says that only an assignment expression is allowed, so the grammar checker
        // will error if it sees a comma expression.
        auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        parseExpected(SyntaxKind::CloseBracketToken);
        return finishNode(factory.createComputedPropertyName(expression), pos);
    }

    auto internPrivateIdentifier(string text) -> string
    {
        privateIdentifiers[text] = text;
        return privateIdentifiers.at(text);
    }

    auto parsePrivateIdentifier() -> Node
    {
        auto pos = getNodePos();
        auto node = factory.createPrivateIdentifier(internPrivateIdentifier(scanner.getTokenText()));
        nextToken();
        return finishNode(node, pos);
    }

    auto parseContextualModifier(SyntaxKind t) -> boolean
    {
        return token() == t && tryParse<boolean>(std::bind(&Parser::nextTokenCanFollowModifier, this));
    }

    auto nextTokenIsOnSameLineAndCanFollowModifier()
    {
        nextToken();
        if (scanner.hasPrecedingLineBreak())
        {
            return false;
        }
        return canFollowModifier();
    }

    auto nextTokenCanFollowModifier() -> boolean
    {
        switch (token())
        {
        case SyntaxKind::ConstKeyword:
            // 'const' is only a modifier if followed by 'enum'.
            return nextToken() == SyntaxKind::EnumKeyword;
        case SyntaxKind::ExportKeyword:
            nextToken();
            if (token() == SyntaxKind::DefaultKeyword)
            {
                return lookAhead<boolean>(std::bind(&Parser::nextTokenCanFollowDefaultKeyword, this));
            }
            if (token() == SyntaxKind::TypeKeyword)
            {
                return lookAhead<boolean>(std::bind(&Parser::nextTokenCanFollowExportModifier, this));
            }
            return canFollowExportModifier();
        case SyntaxKind::DefaultKeyword:
            return nextTokenCanFollowDefaultKeyword();
        case SyntaxKind::StaticKeyword:
            return nextTokenIsOnSameLineAndCanFollowModifier();
        case SyntaxKind::GetKeyword:
        case SyntaxKind::SetKeyword:
            nextToken();
            return canFollowModifier();
        default:
            return nextTokenIsOnSameLineAndCanFollowModifier();
        }
    }

    auto canFollowExportModifier() -> boolean
    {
        return token() != SyntaxKind::AsteriskToken && token() != SyntaxKind::AsKeyword && token() != SyntaxKind::OpenBraceToken &&
               canFollowModifier();
    }

    auto nextTokenCanFollowExportModifier() -> boolean
    {
        nextToken();
        return canFollowExportModifier();
    }

    auto parseAnyContextualModifier() -> boolean
    {
        return isModifierKind(token()) && tryParse<boolean>(std::bind(&Parser::nextTokenCanFollowModifier, this));
    }

    auto canFollowModifier() -> boolean
    {
        return token() == SyntaxKind::OpenBracketToken || token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::AsteriskToken ||
               token() == SyntaxKind::DotDotDotToken || isLiteralPropertyName();
    }

    auto nextTokenCanFollowDefaultKeyword() -> boolean
    {
        nextToken();
        return token() == SyntaxKind::ClassKeyword || token() == SyntaxKind::FunctionKeyword || token() == SyntaxKind::InterfaceKeyword ||
               (token() == SyntaxKind::AbstractKeyword &&
                lookAhead<boolean>(std::bind(&Parser::nextTokenIsClassKeywordOnSameLine, this))) ||
               (token() == SyntaxKind::AsyncKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsFunctionKeywordOnSameLine, this)));
    }

    // True if positioned at the start of a list element
    auto isListElement(ParsingContext parsingContext, boolean inErrorRecovery) -> boolean
    {
        auto node = currentNode(parsingContext);
        if (node)
        {
            return true;
        }

        switch (parsingContext)
        {
        case ParsingContext::SourceElements:
        case ParsingContext::BlockStatements:
        case ParsingContext::SwitchClauseStatements:
            // If we're in error recovery, then we don't want to treat ';'.as<an>() empty statement.
            // The problem is that ';' can show up in far too many contexts, and if we see one
            // and assume it's a statement, then we may bail out inappropriately from whatever
            // we're parsing.  For example, if we have a semicolon in the middle of a class, then
            // we really don't want to assume the class is over and we're on a statement in the
            // outer module.  We just want to consume and move on.
            return !(token() == SyntaxKind::SemicolonToken && inErrorRecovery) && isStartOfStatement();
        case ParsingContext::SwitchClauses:
            return token() == SyntaxKind::CaseKeyword || token() == SyntaxKind::DefaultKeyword;
        case ParsingContext::TypeMembers:
            return lookAhead<boolean>(std::bind(&Parser::isTypeMemberStart, this));
        case ParsingContext::ClassMembers:
            // We allow semicolons.as<class>() elements (as specified by ES6).as<long>().as<we>()'re
            // not in error recovery.  If we're in error recovery, we don't want an errant
            // semicolon to be treated.as<a>() class member (since they're almost always used
            // for statements.
            return lookAhead<boolean>(std::bind(&Parser::isClassMemberStart, this)) ||
                   (token() == SyntaxKind::SemicolonToken && !inErrorRecovery);
        case ParsingContext::EnumMembers:
            // Include open bracket computed properties. This technically also lets in indexers,
            // which would be a candidate for improved error reporting.
            return token() == SyntaxKind::OpenBracketToken || isLiteralPropertyName();
        case ParsingContext::ObjectLiteralMembers:
            switch (token())
            {
            case SyntaxKind::OpenBracketToken:
            case SyntaxKind::AsteriskToken:
            case SyntaxKind::DotDotDotToken:
            case SyntaxKind::DotToken: // Not an object literal member, but don't want to close the object (see
                                       // `tests/cases/fourslash/completionsDotInObjectLiteral.ts`)
                return true;
            default:
                return isLiteralPropertyName();
            }
        case ParsingContext::RestProperties:
            return isLiteralPropertyName();
        case ParsingContext::ObjectBindingElements:
            return token() == SyntaxKind::OpenBracketToken || token() == SyntaxKind::DotDotDotToken || isLiteralPropertyName();
        case ParsingContext::HeritageClauseElement:
            // If we see `{ ... }` then only consume it.as<an>() expression if it is followed by `,` or `{`
            // That way we won't consume the body of a class in its heritage clause.
            if (token() == SyntaxKind::OpenBraceToken)
            {
                return lookAhead<boolean>(std::bind(&Parser::isValidHeritageClauseObjectLiteral, this));
            }

            if (!inErrorRecovery)
            {
                return isStartOfLeftHandSideExpression() && !isHeritageClauseExtendsOrImplementsKeyword();
            }
            else
            {
                // If we're in error recovery we tighten up what we're willing to match.
                // That way we don't treat something like "this".as<a>() valid heritage clause
                // element during recovery.
                return isIdentifier() && !isHeritageClauseExtendsOrImplementsKeyword();
            }
        case ParsingContext::VariableDeclarations:
            return isBindingIdentifierOrPrivateIdentifierOrPattern();
        case ParsingContext::ArrayBindingElements:
            return token() == SyntaxKind::CommaToken || token() == SyntaxKind::DotDotDotToken ||
                   isBindingIdentifierOrPrivateIdentifierOrPattern();
        case ParsingContext::TypeParameters:
            return isIdentifier();
        case ParsingContext::ArrayLiteralMembers:
            switch (token())
            {
            case SyntaxKind::CommaToken:
            case SyntaxKind::DotToken: // Not an array literal member, but don't want to close the array (see
                                       // `tests/cases/fourslash/completionsDotInArrayLiteralInObjectLiteral.ts`)
                return true;
            }
            // falls through
        case ParsingContext::ArgumentExpressions:
            return token() == SyntaxKind::DotDotDotToken || isStartOfExpression();
        case ParsingContext::Parameters:
            return isStartOfParameter(/*isJSDocParameter*/ false);
        case ParsingContext::JSDocParameters:
            return isStartOfParameter(/*isJSDocParameter*/ true);
        case ParsingContext::TypeArguments:
        case ParsingContext::TupleElementTypes:
            return token() == SyntaxKind::CommaToken || isStartOfType();
        case ParsingContext::HeritageClauses:
            return isHeritageClause();
        case ParsingContext::ImportOrExportSpecifiers:
            return scanner.tokenIsIdentifierOrKeyword(token());
        case ParsingContext::JsxAttributes:
            return scanner.tokenIsIdentifierOrKeyword(token()) || token() == SyntaxKind::OpenBraceToken;
        case ParsingContext::JsxChildren:
            return true;
        }

        return Debug::fail<boolean>(S("Non-exhaustive case in 'isListElement'."));
    }

    auto isValidHeritageClauseObjectLiteral() -> boolean
    {
        Debug::_assert(token() == SyntaxKind::OpenBraceToken);
        if (nextToken() == SyntaxKind::CloseBraceToken)
        {
            // if we see "extends {}" then only treat the {}.as<what>() we're extending (and not
            // the class body) if we have:
            //
            //      extends {} {
            //      extends {},
            //      extends {} extends
            //      extends {} implements

            auto next = nextToken();
            return next == SyntaxKind::CommaToken || next == SyntaxKind::OpenBraceToken || next == SyntaxKind::ExtendsKeyword ||
                   next == SyntaxKind::ImplementsKeyword;
        }

        return true;
    }

    auto nextTokenIsIdentifier() -> boolean
    {
        nextToken();
        return isIdentifier();
    }

    auto nextTokenIsIdentifierOrKeyword() -> boolean
    {
        nextToken();
        return scanner.tokenIsIdentifierOrKeyword(token());
    }

    auto nextTokenIsIdentifierOrKeywordOrGreaterThan() -> boolean
    {
        nextToken();
        return scanner.tokenIsIdentifierOrKeywordOrGreaterThan(token());
    }

    auto isHeritageClauseExtendsOrImplementsKeyword() -> boolean
    {
        if (token() == SyntaxKind::ImplementsKeyword || token() == SyntaxKind::ExtendsKeyword)
        {

            return lookAhead<boolean>(std::bind(&Parser::nextTokenIsStartOfExpression, this));
        }

        return false;
    }

    auto nextTokenIsStartOfExpression() -> boolean
    {
        nextToken();
        return isStartOfExpression();
    }

    auto nextTokenIsStartOfType() -> boolean
    {
        nextToken();
        return isStartOfType();
    }

    // True if positioned at a list terminator
    auto isListTerminator(ParsingContext kind) -> boolean
    {
        if (token() == SyntaxKind::EndOfFileToken)
        {
            // Being at the end of the file ends all lists.
            return true;
        }

        switch (kind)
        {
        case ParsingContext::BlockStatements:
        case ParsingContext::SwitchClauses:
        case ParsingContext::TypeMembers:
        case ParsingContext::ClassMembers:
        case ParsingContext::EnumMembers:
        case ParsingContext::ObjectLiteralMembers:
        case ParsingContext::ObjectBindingElements:
        case ParsingContext::ImportOrExportSpecifiers:
            return token() == SyntaxKind::CloseBraceToken;
        case ParsingContext::SwitchClauseStatements:
            return token() == SyntaxKind::CloseBraceToken || token() == SyntaxKind::CaseKeyword || token() == SyntaxKind::DefaultKeyword;
        case ParsingContext::HeritageClauseElement:
            return token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::ExtendsKeyword ||
                   token() == SyntaxKind::ImplementsKeyword;
        case ParsingContext::VariableDeclarations:
            return isVariableDeclaratorListTerminator();
        case ParsingContext::TypeParameters:
            // Tokens other than '>' are here for better error recovery
            return token() == SyntaxKind::GreaterThanToken || token() == SyntaxKind::OpenParenToken ||
                   token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::ExtendsKeyword ||
                   token() == SyntaxKind::ImplementsKeyword;
        case ParsingContext::ArgumentExpressions:
            // Tokens other than ')' are here for better error recovery
            return token() == SyntaxKind::CloseParenToken || token() == SyntaxKind::SemicolonToken;
        case ParsingContext::ArrayLiteralMembers:
        case ParsingContext::TupleElementTypes:
        case ParsingContext::ArrayBindingElements:
            return token() == SyntaxKind::CloseBracketToken;
        case ParsingContext::JSDocParameters:
        case ParsingContext::Parameters:
        case ParsingContext::RestProperties:
            // Tokens other than ')' and ']' (the latter for index signatures) are here for better error recovery
            return token() == SyntaxKind::CloseParenToken ||
                   token() == SyntaxKind::CloseBracketToken /*|| token == SyntaxKind::OpenBraceToken*/;
        case ParsingContext::TypeArguments:
            // All other tokens should cause the type-argument to terminate except comma token
            return token() != SyntaxKind::CommaToken;
        case ParsingContext::HeritageClauses:
            return token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::CloseBraceToken;
        case ParsingContext::JsxAttributes:
            return token() == SyntaxKind::GreaterThanToken || token() == SyntaxKind::SlashToken;
        case ParsingContext::JsxChildren:
            return token() == SyntaxKind::LessThanToken && lookAhead<boolean>(std::bind(&Parser::nextTokenIsSlash, this));
        default:
            return false;
        }
    }

    auto isVariableDeclaratorListTerminator() -> boolean
    {
        // If we can consume a semicolon (either explicitly, or with ASI), then consider us done
        // with parsing the list of variable declarators.
        if (canParseSemicolon())
        {
            return true;
        }

        // in the case where we're parsing the variable declarator of a 'for-in' statement, we
        // are done if we see an 'in' keyword in front of us. Same with for-of
        if (isInOrOfKeyword(token()))
        {
            return true;
        }

        // ERROR RECOVERY TWEAK:
        // For better error recovery, if we see an '=>' then we just stop immediately.  We've got an
        // arrow auto here and it's going to be very unlikely that we'll resynchronize and get
        // another variable declaration.
        if (token() == SyntaxKind::EqualsGreaterThanToken)
        {
            return true;
        }

        // Keep trying to parse out variable declarators.
        return false;
    }

    // True if positioned at element or terminator of the current list or any enclosing list
    auto isInSomeParsingContext() -> boolean
    {
        for (auto kind = (number)ParsingContext::Unknown; kind < (number)ParsingContext::Count; kind++)
        {
            if (!!(parsingContext & (ParsingContext)(1 << kind)))
            {
                if (isListElement((ParsingContext)kind, /*inErrorRecovery*/ true) || isListTerminator((ParsingContext)kind))
                {
                    return true;
                }
            }
        }

        return false;
    }

    // Parses a list of elements
    template <typename T> auto parseList(ParsingContext kind, std::function<T()> parseElement) -> NodeArray<T>
    {
        auto saveParsingContext = parsingContext;
        parsingContext |= (ParsingContext)(1 << (number)kind);
        NodeArray<T> list;
        auto listPos = getNodePos();

        while (!isListTerminator(kind))
        {
            if (isListElement(kind, /*inErrorRecovery*/ false))
            {
                auto element = parseListElement(kind, parseElement);
                list.push_back(element);

                continue;
            }

            if (abortParsingListOrMoveToNextToken(kind))
            {
                break;
            }
        }

        parsingContext = saveParsingContext;
        return createNodeArray(list, listPos);
    }

    template <typename T> auto parseListElement(ParsingContext parsingContext, std::function<T()> parseElement) -> T
    {
        auto node = currentNode(parsingContext);
        if (node)
        {
            return consumeNode(node).as<T>();
        }

        return parseElement();
    }

    auto currentNode(ParsingContext parsingContext) -> Node
    {
        // If we don't have a cursor or the parsing context isn't reusable, there's nothing to reuse.
        //
        // If there is an outstanding parse error that we've encountered, but not attached to
        // some node, then we cannot get a node from the old source tree.  This is because we
        // want to mark the next node we encounter.as<being>() unusable.
        //
        // This Note may be too conservative.  Perhaps we could reuse the node and set the bit
        // on it (or its leftmost child).as<having>() the error.  For now though, being conservative
        // is nice and likely won't ever affect perf.
        if (!syntaxCursor || !isReusableParsingContext(parsingContext) || parseErrorBeforeNextFinishedNode)
        {
            return undefined;
        }

        auto node = ((IncrementalParser::SyntaxCursor &)syntaxCursor).currentNode(scanner.getStartPos());

        // Can't reuse a missing node->
        // Can't reuse a node that intersected the change range.
        // Can't reuse a node that contains a parse error.  This is necessary so that we
        // produce the same set of errors again.
        if (nodeIsMissing(node) || node.intersectsChange || containsParseError(node))
        {
            return undefined;
        }

        // We can only reuse a node if it was parsed under the same strict mode that we're
        // currently in.  i.e. if we originally parsed a node in non-strict mode, but then
        // the user added 'using strict' at the top of the file, then we can't use that node
        // again.as<the>() presence of strict mode may cause us to parse the tokens in the file
        // differently.
        //
        // we Note *can* reuse tokens when the strict mode changes.  That's because tokens
        // are unaffected by strict mode.  It's just the parser will decide what to do with it
        // differently depending on what mode it is in.
        //
        // This also applies to all our other context flags.as<well>().
        auto nodeContextFlags = node->flags & NodeFlags::ContextFlags;
        if (nodeContextFlags != contextFlags)
        {
            return undefined;
        }

        // Ok, we have a node that looks like it could be reused.  Now verify that it is valid
        // in the current list parsing context that we're currently at.
        if (!canReuseNode(node, parsingContext))
        {
            return undefined;
        }

        if (node.as<JSDocContainer>()->jsDocCache.size() > 0)
        {
            // jsDocCache may include tags from parent nodes, which might have been modified.
            node.as<JSDocContainer>()->jsDocCache.clear();
        }

        return node;
    }

    auto consumeNode(Node node) -> Node
    {
        // Move the scanner so it is after the node we just consumed.
        scanner.setTextPos(node->_end);
        nextToken();
        return node;
    }

    auto isReusableParsingContext(ParsingContext parsingContext) -> boolean
    {
        switch (parsingContext)
        {
        case ParsingContext::ClassMembers:
        case ParsingContext::SwitchClauses:
        case ParsingContext::SourceElements:
        case ParsingContext::BlockStatements:
        case ParsingContext::SwitchClauseStatements:
        case ParsingContext::EnumMembers:
        case ParsingContext::TypeMembers:
        case ParsingContext::VariableDeclarations:
        case ParsingContext::JSDocParameters:
        case ParsingContext::Parameters:
            return true;
        }
        return false;
    }

    auto canReuseNode(Node node, ParsingContext parsingContext) -> boolean
    {
        switch (parsingContext)
        {
        case ParsingContext::ClassMembers:
            return isReusableClassMember(node);

        case ParsingContext::SwitchClauses:
            return isReusableSwitchClause(node);

        case ParsingContext::SourceElements:
        case ParsingContext::BlockStatements:
        case ParsingContext::SwitchClauseStatements:
            return isReusableStatement(node);

        case ParsingContext::EnumMembers:
            return isReusableEnumMember(node);

        case ParsingContext::TypeMembers:
            return isReusableTypeMember(node);

        case ParsingContext::VariableDeclarations:
            return isReusableVariableDeclaration(node);

        case ParsingContext::JSDocParameters:
        case ParsingContext::Parameters:
            return isReusableParameter(node);

            // Any other lists we do not care about reusing nodes in.  But feel free to add if
            // you can do so safely.  Danger areas involve nodes that may involve speculative
            // parsing.  If speculative parsing is involved with the node, then the range the
            // parser reached while looking ahead might be in the edited range (see the example
            // in canReuseVariableDeclaratorNode for a good case of this).

            // case ParsingContext::HeritageClauses:
            // This would probably be safe to reuse.  There is no speculative parsing with
            // heritage clauses.

            // case ParsingContext::TypeParameters:
            // This would probably be safe to reuse.  There is no speculative parsing with
            // type parameters.  Note that that's because type *parameters* only occur in
            // unambiguous *type* contexts.  While type *arguments* occur in very ambiguous
            // *expression* contexts.

            // case ParsingContext::TupleElementTypes:
            // This would probably be safe to reuse.  There is no speculative parsing with
            // tuple types.

            // Technically, type argument list types are probably safe to reuse.  While
            // speculative parsing is involved with them (since type argument lists are only
            // produced from speculative parsing a <.as<a>() type argument list), we only have
            // the types because speculative parsing succeeded.  Thus, the lookahead never
            // went past the end of the list and rewound.
            // case ParsingContext::TypeArguments:

            // these Note are almost certainly not safe to ever reuse.  Expressions commonly
            // need a large amount of lookahead, and we should not reuse them.as<they>() may
            // have actually intersected the edit.
            // case ParsingContext::ArgumentExpressions:

            // This is not safe to reuse for the same reason.as<the>() 'AssignmentExpression'
            // cases.  i.e. a property assignment may end with an expression, and thus might
            // have lookahead far beyond it's old node->
            // case ParsingContext::ObjectLiteralMembers:

            // This is probably not safe to reuse.  There can be speculative parsing with
            // type names in a heritage clause.  There can be generic names in the type
            // name list, and there can be left hand side expressions (which can have type
            // arguments.)
            // case ParsingContext::HeritageClauseElement:

            // Perhaps safe to reuse, but it's unlikely we'd see more than a dozen attributes
            // on any given element. Same for children.
            // case ParsingContext::JsxAttributes:
            // case ParsingContext::JsxChildren:
        }

        return false;
    }

    auto isReusableClassMember(Node node) -> boolean
    {
        if (node)
        {
            switch ((SyntaxKind)node)
            {
            case SyntaxKind::Constructor:
            case SyntaxKind::IndexSignature:
            case SyntaxKind::GetAccessor:
            case SyntaxKind::SetAccessor:
            case SyntaxKind::PropertyDeclaration:
            case SyntaxKind::SemicolonClassElement:
                return true;
            case SyntaxKind::MethodDeclaration:
                // Method declarations are not necessarily reusable.  An object-literal
                // may have a method calls "constructor(...)" and we must reparse that
                // into an actual .ConstructorDeclaration.
                auto methodDeclaration = node.as<MethodDeclaration>();
                auto nameIsConstructor = methodDeclaration->name == SyntaxKind::Identifier &&
                                         methodDeclaration->name.as<Identifier>()->originalKeywordKind == SyntaxKind::ConstructorKeyword;

                return !nameIsConstructor;
            }
        }

        return false;
    }

    auto isReusableSwitchClause(Node node) -> boolean
    {
        if (node)
        {
            switch ((SyntaxKind)node)
            {
            case SyntaxKind::CaseClause:
            case SyntaxKind::DefaultClause:
                return true;
            }
        }

        return false;
    }

    auto isReusableStatement(Node node) -> boolean
    {
        if (node)
        {
            switch ((SyntaxKind)node)
            {
            case SyntaxKind::FunctionDeclaration:
            case SyntaxKind::VariableStatement:
            case SyntaxKind::Block:
            case SyntaxKind::IfStatement:
            case SyntaxKind::ExpressionStatement:
            case SyntaxKind::ThrowStatement:
            case SyntaxKind::ReturnStatement:
            case SyntaxKind::SwitchStatement:
            case SyntaxKind::BreakStatement:
            case SyntaxKind::ContinueStatement:
            case SyntaxKind::ForInStatement:
            case SyntaxKind::ForOfStatement:
            case SyntaxKind::ForStatement:
            case SyntaxKind::WhileStatement:
            case SyntaxKind::WithStatement:
            case SyntaxKind::EmptyStatement:
            case SyntaxKind::TryStatement:
            case SyntaxKind::LabeledStatement:
            case SyntaxKind::DoStatement:
            case SyntaxKind::DebuggerStatement:
            case SyntaxKind::ImportDeclaration:
            case SyntaxKind::ImportEqualsDeclaration:
            case SyntaxKind::ExportDeclaration:
            case SyntaxKind::ExportAssignment:
            case SyntaxKind::ModuleDeclaration:
            case SyntaxKind::ClassDeclaration:
            case SyntaxKind::InterfaceDeclaration:
            case SyntaxKind::EnumDeclaration:
            case SyntaxKind::TypeAliasDeclaration:
                return true;
            }
        }

        return false;
    }

    auto isReusableEnumMember(Node node) -> boolean
    {
        return node == SyntaxKind::EnumMember;
    }

    auto isReusableTypeMember(Node node) -> boolean
    {
        if (node)
        {
            switch ((SyntaxKind)node)
            {
            case SyntaxKind::ConstructSignature:
            case SyntaxKind::MethodSignature:
            case SyntaxKind::IndexSignature:
            case SyntaxKind::PropertySignature:
            case SyntaxKind::CallSignature:
                return true;
            }
        }

        return false;
    }

    auto isReusableVariableDeclaration(Node node) -> boolean
    {
        if (node != SyntaxKind::VariableDeclaration)
        {
            return false;
        }

        // Very subtle incremental parsing bug.  Consider the following code:
        //
        //      auto v = new List < A, B
        //
        // This is actually legal code.  It's a list of variable declarators "v = new List<A"
        // on one side and "B" on the other. If you then change that to:
        //
        //      auto v = new List < A, B >()
        //
        // then we have a problem.  "v = new List<A" doesn't intersect the change range, so we
        // start reparsing at "B" and we completely fail to handle this properly.
        //
        // In order to prevent this, we do not allow a variable declarator to be reused if it
        // has an initializer.
        auto variableDeclarator = node.as<VariableDeclaration>();
        return variableDeclarator->initializer == undefined;
    }

    auto isReusableParameter(Node node) -> boolean
    {
        if (node != SyntaxKind::Parameter)
        {
            return false;
        }

        // See the comment in isReusableVariableDeclaration for why we do this.
        auto parameter = node.as<ParameterDeclaration>();
        return parameter->initializer == undefined;
    }

    // Returns true if we should abort parsing.
    auto abortParsingListOrMoveToNextToken(ParsingContext kind) -> boolean
    {
        parsingContextErrors(kind);
        if (isInSomeParsingContext())
        {
            return true;
        }

        nextToken();
        return false;
    }

    auto parsingContextErrors(ParsingContext context) -> void
    {
        switch (context)
        {
        case ParsingContext::SourceElements:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Declaration_or_statement_expected));
        case ParsingContext::BlockStatements:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Declaration_or_statement_expected));
        case ParsingContext::SwitchClauses:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::case_or_default_expected));
        case ParsingContext::SwitchClauseStatements:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Statement_expected));
        case ParsingContext::RestProperties: // fallthrough
        case ParsingContext::TypeMembers:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Property_or_signature_expected));
        case ParsingContext::ClassMembers:
            return parseErrorAtCurrentToken(
                data::DiagnosticMessage(Diagnostics::Unexpected_token_A_constructor_method_accessor_or_property_was_expected));
        case ParsingContext::EnumMembers:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Enum_member_expected));
        case ParsingContext::HeritageClauseElement:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Expression_expected));
        case ParsingContext::VariableDeclarations:
            return isKeyword(token())
                       ? parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::_0_is_not_allowed_as_a_variable_declaration_name),
                                                  scanner.tokenToString(token()))
                       : parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Variable_declaration_expected));
        case ParsingContext::ObjectBindingElements:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Property_destructuring_pattern_expected));
        case ParsingContext::ArrayBindingElements:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Array_element_destructuring_pattern_expected));
        case ParsingContext::ArgumentExpressions:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Argument_expression_expected));
        case ParsingContext::ObjectLiteralMembers:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Property_assignment_expected));
        case ParsingContext::ArrayLiteralMembers:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Expression_or_comma_expected));
        case ParsingContext::JSDocParameters:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Parameter_declaration_expected));
        case ParsingContext::Parameters:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Parameter_declaration_expected));
        case ParsingContext::TypeParameters:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Type_parameter_declaration_expected));
        case ParsingContext::TypeArguments:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Type_argument_expected));
        case ParsingContext::TupleElementTypes:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Type_expected));
        case ParsingContext::HeritageClauses:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Unexpected_token_expected));
        case ParsingContext::ImportOrExportSpecifiers:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Identifier_expected));
        case ParsingContext::JsxAttributes:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Identifier_expected));
        case ParsingContext::JsxChildren:
            return parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::Identifier_expected));
            return; // GH TODO#18217 `Debug::_assertNever default(context);`
        }
    }

    // Parses a comma-delimited list of elements
    template <typename T>
    auto parseDelimitedList(ParsingContext kind, std::function<T()> parseElement, boolean considerSemicolonAsDelimiter = false)
        -> NodeArray<T>
    {
        auto saveParsingContext = parsingContext;
        parsingContext |= (ParsingContext)(1 << (number)kind);
        NodeArray<T> list;
        auto listPos = getNodePos();

        auto commaStart = -1; // Meaning the previous token was not a comma
        while (true)
        {
            if (isListElement(kind, /*inErrorRecovery*/ false))
            {
                auto startPos = scanner.getStartPos();
                list.push_back(parseListElement<T>(kind, parseElement));
                commaStart = scanner.getTokenPos();

                if (parseOptional(SyntaxKind::CommaToken))
                {
                    // No need to check for a zero length node since we know we parsed a comma
                    continue;
                }

                commaStart = -1; // Back to the state where the last token was not a comma
                if (isListTerminator(kind))
                {
                    break;
                }

                // We didn't get a comma, and the list wasn't terminated, explicitly parse
                // out a comma so we give a good error message.
                parseExpected(SyntaxKind::CommaToken, getExpectedCommaDiagnostic(kind));

                // If the token was a semicolon, and the caller allows that, then skip it and
                // continue.  This ensures we get back on track and don't result in tons of
                // parse errors.  For example, this can happen when people do things like use
                // a semicolon to delimit object literal members.   we Note'll have already
                // reported an error when we called parseExpected above.
                if (considerSemicolonAsDelimiter && token() == SyntaxKind::SemicolonToken && !scanner.hasPrecedingLineBreak())
                {
                    nextToken();
                }
                if (startPos == scanner.getStartPos())
                {
                    // What we're parsing isn't actually remotely recognizable.as<a>() element and we've consumed no tokens whatsoever
                    // Consume a token to advance the parser in some way and avoid an infinite loop
                    // This can happen when we're speculatively parsing parenthesized expressions which we think may be arrow functions,
                    // or when a modifier keyword which is disallowed.as<a>() parameter name (ie, `static` in strict mode) is supplied
                    nextToken();
                }
                continue;
            }

            if (isListTerminator(kind))
            {
                break;
            }

            if (abortParsingListOrMoveToNextToken(kind))
            {
                break;
            }
        }

        parsingContext = saveParsingContext;
        // Recording the trailing comma is deliberately done after the previous
        // loop, and not just if we see a list terminator. This is because the list
        // may have ended incorrectly, but it is still important to know if there
        // was a trailing comma.
        // Check if the last token was a comma.
        // Always preserve a trailing comma by marking it on the NodeArray
        return createNodeArray<T>(list, listPos, /*end*/ -1, commaStart >= 0);
    }

    auto getExpectedCommaDiagnostic(ParsingContext kind) -> DiagnosticMessage
    {
        return kind == ParsingContext::EnumMembers
                   ? DiagnosticMessage(data::DiagnosticMessage(Diagnostics::An_enum_member_name_must_be_followed_by_a_or))
                   : undefined;
    }

    template <typename T> using MissingList = NodeArray<T>;

    template <typename T> auto createMissingList() -> MissingList<T>
    {
        auto list = createNodeArray<T>(NodeArray<T>(), getNodePos());
        list.isMissingList = true;
        return list;
    }

    template <typename T> auto isMissingList(NodeArray<T> arr) -> boolean
    {
        return arr.isMissingList;
    }

    template <typename T>
    auto parseBracketedList(ParsingContext kind, std::function<T()> parseElement, SyntaxKind open, SyntaxKind close) -> NodeArray<T>
    {
        if (parseExpected(open))
        {
            auto result = parseDelimitedList(kind, parseElement);
            parseExpected(close);
            return result;
        }

        return createMissingList<T>();
    }

    auto parseEntityName(boolean allowReservedWords, DiagnosticMessage diagnosticMessage = undefined) -> Identifier
    {
        auto pos = getNodePos();
        auto entity = allowReservedWords ? parseIdentifierName(diagnosticMessage) : parseIdentifier(diagnosticMessage);
        auto dotPos = getNodePos();
        while (parseOptional(SyntaxKind::DotToken))
        {
            if (token() == SyntaxKind::LessThanToken)
            {
                // the entity is part of a JSDoc-style generic, so record the trailing dot for later error reporting
                entity->jsdocDotPos = dotPos;
                break;
            }
            dotPos = getNodePos();
            entity = finishNode(factory.createQualifiedName(
                                    entity, parseRightSideOfDot(allowReservedWords, /* allowPrivateIdentifiers */ false).as<Identifier>()),
                                pos);
        }
        return entity;
    }

    auto createQualifiedName(EntityName entity, Identifier name) -> QualifiedName
    {
        return finishNode(factory.createQualifiedName(entity, name), entity->pos).as<QualifiedName>();
    }

    auto parseRightSideOfDot(boolean allowIdentifierNames, boolean allowPrivateIdentifiers) -> Node
    {
        // Technically a keyword is valid here.as<all>() identifiers and keywords are identifier names.
        // However, often we'll encounter this in error situations when the identifier or keyword
        // is actually starting another valid construct.
        //
        // So, we check for the following specific case:
        //
        //      name.
        //      identifierOrKeyword identifierNameOrKeyword
        //
        // the Note newlines are important here.  For example, if that above code
        // were rewritten into:
        //
        //      name.identifierOrKeyword
        //      identifierNameOrKeyword
        //
        // Then we would consider it valid.  That's because ASI would take effect and
        // the code would be implicitly: "name.identifierOrKeyword; identifierNameOrKeyword".
        // In the first case though, ASI will not take effect because there is not a
        // line terminator after the identifier or keyword.
        if (scanner.hasPrecedingLineBreak() && scanner.tokenIsIdentifierOrKeyword(token()))
        {
            auto matchesPattern = lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOnSameLine, this));

            if (matchesPattern)
            {
                // Report that we need an identifier.  However, report it right after the dot,
                // and not on the next token.  This is because the next token might actually
                // be an identifier and the error would be quite confusing.
                return createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ true,
                                                     data::DiagnosticMessage(Diagnostics::Identifier_expected));
            }
        }

        if (token() == SyntaxKind::PrivateIdentifier)
        {
            auto node = parsePrivateIdentifier();
            return allowPrivateIdentifiers ? node
                                           : createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ true,
                                                                           data::DiagnosticMessage(Diagnostics::Identifier_expected));
        }

        return allowIdentifierNames ? parseIdentifierName() : parseIdentifier();
    }

    auto parseTemplateSpans(boolean isTaggedTemplate) -> NodeArray<TemplateSpan>
    {
        auto pos = getNodePos();
        NodeArray<TemplateSpan> list;
        TemplateSpan node;
        do
        {
            node = parseTemplateSpan(isTaggedTemplate);
            list.push_back(node);
        } while (node->literal == SyntaxKind::TemplateMiddle);
        return createNodeArray(list, pos);
    }

    auto parseTemplateExpression(boolean isTaggedTemplate) -> TemplateExpression
    {
        auto pos = getNodePos();
        auto head = parseTemplateHead(isTaggedTemplate);
        auto spans = parseTemplateSpans(isTaggedTemplate);
        return finishNode(factory.createTemplateExpression(head, spans), pos);
    }

    auto parseTemplateType() -> TemplateLiteralTypeNode
    {
        auto pos = getNodePos();
        auto head = parseTemplateHead(/*isTaggedTemplate*/ false);
        auto spans = parseTemplateTypeSpans();
        return finishNode(factory.createTemplateLiteralType(head, spans), pos);
    }

    auto parseTemplateTypeSpans() -> NodeArray<TemplateLiteralTypeSpan>
    {
        auto pos = getNodePos();
        NodeArray<TemplateLiteralTypeSpan> list;
        TemplateLiteralTypeSpan node;
        do
        {
            node = parseTemplateTypeSpan();
            list.push_back(node);
        } while (node->literal == SyntaxKind::TemplateMiddle);
        return createNodeArray(list, pos);
    }

    auto parseTemplateTypeSpan() -> TemplateLiteralTypeSpan
    {
        auto pos = getNodePos();
        auto type = parseType();
        auto span = parseLiteralOfTemplateSpan(/*isTaggedTemplate*/ false);
        return finishNode(factory.createTemplateLiteralTypeSpan(type, span), pos);
    }

    auto parseLiteralOfTemplateSpan(boolean isTaggedTemplate) -> Node
    {
        if (token() == SyntaxKind::CloseBraceToken)
        {
            reScanTemplateToken(isTaggedTemplate);
            return parseTemplateMiddleOrTemplateTail();
        }
        else
        {
            // TODO(rbuckton) -> Do we need to call `parseExpectedToken` or can we just call `createMissingNode` directly?
            return parseExpectedToken(SyntaxKind::TemplateTail, data::DiagnosticMessage(Diagnostics::_0_expected),
                                      scanner.tokenToString(SyntaxKind::CloseBraceToken));
        }
    }

    auto parseTemplateSpan(boolean isTaggedTemplate) -> TemplateSpan
    {
        auto pos = getNodePos();
        auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        auto span = parseLiteralOfTemplateSpan(isTaggedTemplate);
        return finishNode(factory.createTemplateSpan(expression, span), pos);
    }

    auto parseLiteralNode() -> LiteralExpression
    {
        return parseLiteralLikeNode(token()).as<LiteralExpression>();
    }

    auto parseTemplateHead(boolean isTaggedTemplate) -> TemplateHead
    {
        if (isTaggedTemplate)
        {
            reScanTemplateHeadOrNoSubstitutionTemplate();
        }
        auto fragment = parseLiteralLikeNode(token());
        Debug::_assert(fragment == SyntaxKind::TemplateHead, S("Template head has wrong token kind"));
        return fragment.as<TemplateHead>();
    }

    auto parseTemplateMiddleOrTemplateTail() -> Node
    {
        auto fragment = parseLiteralLikeNode(token());
        Debug::_assert(fragment == SyntaxKind::TemplateMiddle || fragment == SyntaxKind::TemplateTail,
                       S("Template fragment has wrong token kind"));
        return fragment;
    }

    auto getTemplateLiteralRawText(SyntaxKind kind) -> string
    {
        auto isLast = kind == SyntaxKind::NoSubstitutionTemplateLiteral || kind == SyntaxKind::TemplateTail;
        auto tokenText = scanner.getTokenText();
        return safe_string(tokenText).substring(1, tokenText.size() - (scanner.isUnterminated() ? 0 : isLast ? 1 : 2));
    }

    auto parseLiteralLikeNode(SyntaxKind kind) -> LiteralLikeNode
    {
        auto pos = getNodePos();
        LiteralLikeNode node =
            isTemplateLiteralKind(kind) ? factory
                                              .createTemplateLiteralLikeNode(kind, scanner.getTokenValue(), getTemplateLiteralRawText(kind),
                                                                             scanner.getTokenFlags() & TokenFlags::TemplateLiteralLikeFlags)
                                              .as<LiteralLikeNode>()
                                        :
                                        // Octal literals are not allowed in strict mode or ES5
                                        // Note that theoretically the following condition would hold true literals like 009,
                                        // which is not octal. But because of how the scanner separates the tokens, we would
                                        // never get a token like this. Instead, we would get 00 and 9.as<two>() separate tokens.
                                        // We also do not need to check for negatives because any prefix operator would be part of a
                                        // parent unary expression.
                kind == SyntaxKind::NumericLiteral
                ? factory.createNumericLiteral(scanner.getTokenValue(), scanner.getNumericLiteralFlags()).as<LiteralLikeNode>()
            : kind == SyntaxKind::StringLiteral ? factory
                                                      .createStringLiteral(scanner.getTokenValue(), /*isSingleQuote*/ /*undefined*/ false,
                                                                           scanner.hasExtendedUnicodeEscape())
                                                      .as<LiteralLikeNode>()
            : isLiteralKind(kind) ? factory.createLiteralLikeNode(kind, scanner.getTokenValue())
                                  : Debug::fail<LiteralLikeNode>();

        if (scanner.hasExtendedUnicodeEscape())
        {
            node->hasExtendedUnicodeEscape = true;
        }

        if (scanner.isUnterminated())
        {
            node->isUnterminated = true;
        }

        nextToken();
        return finishNode(node, pos);
    }

    // TYPES

    auto parseEntityNameOfTypeReference() -> Node
    {
        return parseEntityName(/*allowReservedWords*/ true, data::DiagnosticMessage(Diagnostics::Type_expected));
    }

    auto parseTypeArgumentsOfTypeReference() -> NodeArray<TypeNode>
    {
        if (!scanner.hasPrecedingLineBreak() && reScanLessThanToken() == SyntaxKind::LessThanToken)
        {
            return parseBracketedList<TypeNode>(ParsingContext::TypeArguments, std::bind(&Parser::parseType, this),
                                                SyntaxKind::LessThanToken, SyntaxKind::GreaterThanToken);
        }

        return undefined;
    }

    auto parseTypeReference() -> TypeReferenceNode
    {
        auto pos = getNodePos();
        auto entityNameOfTypeReference = parseEntityNameOfTypeReference();
        auto typeArgumentsOfTypeReference = parseTypeArgumentsOfTypeReference();
        return finishNode(factory.createTypeReferenceNode(entityNameOfTypeReference, typeArgumentsOfTypeReference), pos);
    }

    // If true, we should abort parsing an error function.
    auto typeHasArrowFunctionBlockingParseError(TypeNode node) -> boolean
    {
        switch ((SyntaxKind)node)
        {
        case SyntaxKind::TypeReference:
            return nodeIsMissing(node.as<TypeReferenceNode>()->typeName);
        case SyntaxKind::FunctionType:
        case SyntaxKind::ConstructorType: {
            auto res = node.as<FunctionOrConstructorTypeNode>();
            if (res == SyntaxKind::FunctionType)
            {
                auto res1 = res.as<FunctionTypeNode>();
                return isMissingList(res1->parameters) || typeHasArrowFunctionBlockingParseError(res1->type);
            }
            else
            {
                auto res1 = res.as<ConstructorTypeNode>();
                return isMissingList(res1->parameters) || typeHasArrowFunctionBlockingParseError(res1->type);
            }
        }
        case SyntaxKind::ParenthesizedType:
            return typeHasArrowFunctionBlockingParseError(node.as<ParenthesizedTypeNode>()->type);
        default:
            return false;
        }
    }

    auto parseThisTypePredicate(ThisTypeNode lhs) -> TypePredicateNode
    {
        nextToken();
        return finishNode(factory.createTypePredicateNode(/*assertsModifier*/ undefined, lhs, parseType()), lhs->pos);
    }

    auto parseThisTypeNode() -> ThisTypeNode
    {
        auto pos = getNodePos();
        nextToken();
        return finishNode(factory.createThisTypeNode(), pos);
    }

    auto parseJSDocAllType() -> Node
    {
        auto pos = getNodePos();
        nextToken();
        return finishNode(factory.createJSDocAllType(), pos);
    }

    auto parseJSDocNonNullableType() -> TypeNode
    {
        auto pos = getNodePos();
        nextToken();
        return finishNode(factory.createJSDocNonNullableType(parseNonArrayType()), pos);
    }

    auto parseJSDocUnknownOrNullableType() -> Node
    {
        auto pos = getNodePos();
        // skip the ?
        nextToken();

        // Need to lookahead to decide if this is a nullable or unknown type.

        // Here are cases where we'll pick the unknown type:
        //
        //      Foo(?,
        //      { a: ? }
        //      Foo(?)
        //      Foo<?>
        //      Foo(?=
        //      (?|
        if (token() == SyntaxKind::CommaToken || token() == SyntaxKind::CloseBraceToken || token() == SyntaxKind::CloseParenToken ||
            token() == SyntaxKind::GreaterThanToken || token() == SyntaxKind::EqualsToken || token() == SyntaxKind::BarToken)
        {
            return finishNode(factory.createJSDocUnknownType(), pos);
        }
        else
        {
            return finishNode(factory.createJSDocNullableType(parseType()), pos);
        }
    }

    auto parseJSDocFunctionType() -> Node
    {
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        if (lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParen, this)))
        {
            nextToken();
            auto parameters = parseParameters(SignatureFlags::Type | SignatureFlags::JSDoc);
            auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
            return withJSDoc(finishNode(factory.createJSDocFunctionType(parameters, type), pos), hasJSDoc);
        }
        return finishNode(factory.createTypeReferenceNode(parseIdentifierName(), /*typeArguments*/ undefined), pos);
    }

    auto parseJSDocParameter() -> ParameterDeclaration
    {
        auto pos = getNodePos();
        Identifier name;
        if (token() == SyntaxKind::ThisKeyword || token() == SyntaxKind::NewKeyword)
        {
            name = parseIdentifierName();
            parseExpected(SyntaxKind::ColonToken);
        }
        return finishNode(
            factory.createParameterDeclaration(
                /*decorators*/ undefined,
                /*modifiers*/ undefined,
                /*dotDotDotToken*/ undefined,
                // TODO(rbuckton) -> JSDoc parameters don't have names (except `this`/`new`), should we manufacture an empty identifier?
                name,
                /*questionToken*/ undefined, parseJSDocType(),
                /*initializer*/ undefined),
            pos);
    }

    auto parseJSDocType() -> TypeNode
    {
        scanner.setInJSDocType(true);
        auto pos = getNodePos();
        if (parseOptional(SyntaxKind::ModuleKeyword))
        {
            // TODO(rbuckton) -> We never set the type for a JSDocNamepathType. What should we put here?
            auto moduleTag = factory.createJSDocNamepathType(/*type*/ undefined);
            while (true)
            {
                switch (token())
                {
                case SyntaxKind::CloseBraceToken:
                case SyntaxKind::EndOfFileToken:
                case SyntaxKind::CommaToken:
                case SyntaxKind::WhitespaceTrivia:
                    break;
                default:
                    nextTokenJSDoc();
                }
            }

            scanner.setInJSDocType(false);
            return finishNode(moduleTag, pos);
        }

        auto hasDotDotDot = parseOptional(SyntaxKind::DotDotDotToken);
        auto type = parseTypeOrTypePredicate();
        scanner.setInJSDocType(false);
        if (hasDotDotDot)
        {
            type = finishNode(factory.createJSDocVariadicType(type), pos);
        }
        if (token() == SyntaxKind::EqualsToken)
        {
            nextToken();
            return finishNode(factory.createJSDocOptionalType(type), pos);
        }
        return type;
    }

    auto parseTypeQuery() -> TypeQueryNode
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::TypeOfKeyword);
        return finishNode(factory.createTypeQueryNode(parseEntityName(/*allowReservedWords*/ true)), pos);
    }

    auto parseTypeParameter() -> TypeParameterDeclaration
    {
        auto pos = getNodePos();
        auto name = parseIdentifier();
        TypeNode constraint;
        Expression expression;
        if (parseOptional(SyntaxKind::ExtendsKeyword))
        {
            // It's not uncommon for people to write improper constraints to a generic.  If the
            // user writes a constraint that is an expression and not an actual type, then parse
            // it out.as<an>() expression (so we can recover well), but report that a type is needed
            // instead.
            if (isStartOfType() || !isStartOfExpression())
            {
                constraint = parseType();
            }
            else
            {
                // It was not a type, and it looked like an expression.  Parse out an expression
                // here so we recover well.  it Note is important that we call parseUnaryExpression
                // and not parseExpression here.  If the user has:
                //
                //      <T extends string()>
                //
                // We do *not* want to consume the `>`.as<we>()'re consuming the expression for string().
                expression = parseUnaryExpressionOrHigher();
            }
        }

        auto defaultType = parseOptional(SyntaxKind::EqualsToken) ? parseType() : undefined;
        auto node = factory.createTypeParameterDeclaration(name, constraint, defaultType);
        node->expression = expression;
        return finishNode(node, pos);
    }

    auto parseTypeParameters() -> NodeArray<TypeParameterDeclaration>
    {
        if (token() == SyntaxKind::LessThanToken)
        {
            return parseBracketedList<TypeParameterDeclaration>(ParsingContext::TypeParameters,
                                                                std::bind(&Parser::parseTypeParameter, this), SyntaxKind::LessThanToken,
                                                                SyntaxKind::GreaterThanToken);
        }

        return undefined;
    }

    auto isStartOfParameter(boolean isJSDocParameter) -> boolean
    {
        return token() == SyntaxKind::DotDotDotToken || isBindingIdentifierOrPrivateIdentifierOrPattern() || isModifierKind(token()) ||
               token() == SyntaxKind::AtToken || isStartOfType(/*inStartOfParameter*/ !isJSDocParameter);
    }

    auto parseNameOfParameter(ModifiersArray modifiers)
    {
        // FormalParameter [Yield,Await]:
        //      BindingElement[?Yield,?Await]
        auto name = parseIdentifierOrPattern(data::DiagnosticMessage(Diagnostics::Private_identifiers_cannot_be_used_as_parameters));
        if (getFullWidth(name) == 0 && !some<ModifiersArray>(modifiers) && isModifierKind(token()))
        {
            // in cases like
            // 'use strict'
            // auto foo(static)
            // isParameter('static') == true, because of isModifier('static')
            // however 'static' is not a legal identifier in a strict mode.
            // so result of this auto will be ParameterDeclaration (flags = 0, name = missing, type = undefined, initializer = undefined)
            // and current token will not change => parsing of the enclosing parameter list will last till the end of time (or OOM)
            // to avoid this we'll advance cursor to the next token.
            nextToken();
        }
        return name;
    }

    auto parseParameterInOuterAwaitContext() -> ParameterDeclaration
    {
        return parseParameterWorker(/*inOuterAwaitContext*/ true);
    }

    auto parseParameter() -> ParameterDeclaration
    {
        return parseParameterWorker(/*inOuterAwaitContext*/ false);
    }

    auto parseParameterWorker(boolean inOuterAwaitContext) -> ParameterDeclaration
    {
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        if (token() == SyntaxKind::ThisKeyword)
        {
            auto identifier = createIdentifier(/*isIdentifier*/ true);
            auto typeAnnotation = parseTypeAnnotation();
            auto node = factory.createParameterDeclaration(
                /*decorators*/ undefined,
                /*modifiers*/ undefined,
                /*dotDotDotToken*/ undefined, identifier,
                /*questionToken*/ undefined, typeAnnotation,
                /*initializer*/ undefined);
            return withJSDoc(finishNode(node, pos), hasJSDoc);
        }

        // FormalParameter [Yield,Await]:
        //      BindingElement[?Yield,?Await]

        // Decorators are parsed in the outer [Await] context, the rest of the parameter is parsed in the function's [Await] context->
        auto decorators =
            inOuterAwaitContext ? doInAwaitContext<NodeArray<Decorator>>(std::bind(&Parser::parseDecorators, this)) : parseDecorators();
        auto savedTopLevel = topLevel;
        topLevel = false;
        auto modifiers = parseModifiers();

        auto dotDotDotToken = parseOptionalToken(SyntaxKind::DotDotDotToken);
        auto nameOfParameter = parseNameOfParameter(modifiers);
        auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
        auto typeAnnotation = parseTypeAnnotation();
        auto initializer = parseInitializer();

        auto node = withJSDoc(finishNode(factory.createParameterDeclaration(decorators, modifiers, dotDotDotToken, nameOfParameter,
                                                                            questionToken, typeAnnotation, initializer),
                                         pos),
                              hasJSDoc);
        topLevel = savedTopLevel;
        return node;
    }

    auto parseReturnType(SyntaxKind returnToken, boolean isType) -> TypeNode
    {
        if (shouldParseReturnType(returnToken, isType))
        {
            return parseTypeOrTypePredicate();
        }

        return undefined;
    }

    auto shouldParseReturnType(SyntaxKind returnToken, boolean isType) -> boolean
    {
        if (returnToken == SyntaxKind::EqualsGreaterThanToken)
        {
            parseExpected(returnToken);
            return true;
        }
        else if (parseOptional(SyntaxKind::ColonToken))
        {
            return true;
        }
        else if (isType && token() == SyntaxKind::EqualsGreaterThanToken)
        {
            // This is easy to get backward, especially in type contexts, so parse the type anyway
            parseErrorAtCurrentToken(data::DiagnosticMessage(Diagnostics::_0_expected), scanner.tokenToString(SyntaxKind::ColonToken));
            nextToken();
            return true;
        }
        return false;
    }

    auto parseParametersWorker(SignatureFlags flags)
    {
        // FormalParameters [Yield,Await]: (modified)
        //      [empty]
        //      FormalParameterList[?Yield,Await]
        //
        // FormalParameter[Yield,Await]: (modified)
        //      BindingElement[?Yield,Await]
        //
        // BindingElement [Yield,Await]: (modified)
        //      SingleNameBinding[?Yield,?Await]
        //      BindingPattern[?Yield,?Await]Initializer [In, ?Yield,?Await] opt
        //
        // SingleNameBinding [Yield,Await]:
        //      BindingIdentifier[?Yield,?Await]Initializer [In, ?Yield,?Await] opt
        auto savedYieldContext = inYieldContext();
        auto savedAwaitContext = inAwaitContext();

        setYieldContext(!!(flags & SignatureFlags::Yield));
        setAwaitContext(!!(flags & SignatureFlags::Await));

        auto parameters =
            !!(flags & SignatureFlags::JSDoc)
                ? parseDelimitedList<ParameterDeclaration>(ParsingContext::JSDocParameters, std::bind(&Parser::parseJSDocParameter, this))
                : parseDelimitedList<ParameterDeclaration>(ParsingContext::Parameters,
                                                           savedAwaitContext ? std::bind(&Parser::parseParameterInOuterAwaitContext, this)
                                                                             : std::bind(&Parser::parseParameter, this));

        setYieldContext(savedYieldContext);
        setAwaitContext(savedAwaitContext);

        return parameters;
    }

    auto parseParameters(SignatureFlags flags) -> NodeArray<ParameterDeclaration>
    {
        // FormalParameters [Yield,Await]: (modified)
        //      [empty]
        //      FormalParameterList[?Yield,Await]
        //
        // FormalParameter[Yield,Await]: (modified)
        //      BindingElement[?Yield,Await]
        //
        // BindingElement [Yield,Await]: (modified)
        //      SingleNameBinding[?Yield,?Await]
        //      BindingPattern[?Yield,?Await]Initializer [In, ?Yield,?Await] opt
        //
        // SingleNameBinding [Yield,Await]:
        //      BindingIdentifier[?Yield,?Await]Initializer [In, ?Yield,?Await] opt
        if (!parseExpected(SyntaxKind::OpenParenToken))
        {
            return createMissingList<ParameterDeclaration>();
        }

        auto parameters = parseParametersWorker(flags);
        parseExpected(SyntaxKind::CloseParenToken);
        return parameters;
    }

    auto parseTypeMemberSemicolon()
    {
        // We allow type members to be separated by commas or (possibly ASI) semicolons.
        // First check if it was a comma.  If so, we're done with the member.
        if (parseOptional(SyntaxKind::CommaToken))
        {
            return;
        }

        // Didn't have a comma.  We must have a (possible ASI) semicolon.
        parseSemicolon();
    }

    auto parseSignatureMember(SyntaxKind kind) -> Node
    {
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        if (kind == SyntaxKind::ConstructSignature)
        {
            parseExpected(SyntaxKind::NewKeyword);
        }

        auto typeParameters = parseTypeParameters();
        auto parameters = parseParameters(SignatureFlags::Type);
        auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ true);
        parseTypeMemberSemicolon();
        auto node = kind == SyntaxKind::CallSignature
                        ? factory.createCallSignature(typeParameters, parameters, type).as<SignatureDeclarationBase>()
                        : factory.createConstructSignature(typeParameters, parameters, type).as<SignatureDeclarationBase>();
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto isIndexSignature() -> boolean
    {
        return token() == SyntaxKind::OpenBracketToken && lookAhead<boolean>(std::bind(&Parser::isUnambiguouslyIndexSignature, this));
    }

    auto isUnambiguouslyIndexSignature() -> boolean
    {
        // The only allowed sequence is:
        //
        //   [id:
        //
        // However, for error recovery, we also check the following cases:
        //
        //   [...
        //   [id,
        //   [id?,
        //   [id?:
        //   [id?]
        //   [public id
        //   [private id
        //   [protected id
        //   []
        //
        nextToken();
        if (token() == SyntaxKind::DotDotDotToken || token() == SyntaxKind::CloseBracketToken)
        {
            return true;
        }

        if (isModifierKind(token()))
        {
            nextToken();
            if (isIdentifier())
            {
                return true;
            }
        }
        else if (!isIdentifier())
        {
            return false;
        }
        else
        {
            // Skip the identifier
            nextToken();
        }

        // A colon signifies a well formed indexer
        // A comma should be a badly formed indexer because comma expressions are not allowed
        // in computed properties.
        if (token() == SyntaxKind::ColonToken || token() == SyntaxKind::CommaToken)
        {
            return true;
        }

        // Question mark could be an indexer with an optional property,
        // or it could be a conditional expression in a computed property.
        if (token() != SyntaxKind::QuestionToken)
        {
            return false;
        }

        // If any of the following tokens are after the question mark, it cannot
        // be a conditional expression, so treat it.as<an>() indexer.
        nextToken();
        return token() == SyntaxKind::ColonToken || token() == SyntaxKind::CommaToken || token() == SyntaxKind::CloseBracketToken;
    }

    auto parseIndexSignatureDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> IndexSignatureDeclaration
    {
        auto parameters = parseBracketedList<ParameterDeclaration>(ParsingContext::Parameters, std::bind(&Parser::parseParameter, this),
                                                                   SyntaxKind::OpenBracketToken, SyntaxKind::CloseBracketToken);
        auto type = parseTypeAnnotation();
        parseTypeMemberSemicolon();
        auto node = factory.createIndexSignature(decorators, modifiers, parameters, type);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parsePropertyOrMethodSignature(number pos, boolean hasJSDoc, NodeArray<Modifier> modifiers) -> Node
    {
        auto name = parsePropertyName();
        auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
        Node node;
        if (token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken)
        {
            // Method signatures don't exist in expression contexts.  So they have neither
            // [Yield] nor [Await]
            auto typeParameters = parseTypeParameters();
            auto parameters = parseParameters(SignatureFlags::Type);
            auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ true);
            node = factory.createMethodSignature(modifiers, name, questionToken, typeParameters, parameters, type);
        }
        else
        {
            auto type = parseTypeAnnotation();
            node = factory.createPropertySignature(modifiers, name, questionToken, type);
            // Although type literal properties cannot not have initializers, we attempt
            // to parse an initializer so we can report in the checker that an interface
            // property or type literal property cannot have an initializer.
            if (token() == SyntaxKind::EqualsToken)
                node.as<PropertySignature>()->initializer = parseInitializer();
        }
        parseTypeMemberSemicolon();
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto isTypeMemberStart() -> boolean
    {
        // Return true if we have the start of a signature member
        if (token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken)
        {
            return true;
        }
        auto idToken = false;
        // Eat up all modifiers, but hold on to the last one in case it is actually an identifier
        while (isModifierKind(token()))
        {
            idToken = true;
            nextToken();
        }
        // Index signatures and computed property names are type members
        if (token() == SyntaxKind::OpenBracketToken)
        {
            return true;
        }
        // Try to get the first property-like token following all modifiers
        if (isLiteralPropertyName())
        {
            idToken = true;
            nextToken();
        }
        // If we were able to get any potential identifier, check that it is
        // the start of a member declaration
        if (idToken)
        {
            return token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken || token() == SyntaxKind::QuestionToken ||
                   token() == SyntaxKind::ColonToken || token() == SyntaxKind::CommaToken || canParseSemicolon();
        }
        return false;
    }

    auto parseTypeMember() -> TypeElement
    {
        if (token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken)
        {
            return parseSignatureMember(SyntaxKind::CallSignature);
        }
        if (token() == SyntaxKind::NewKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParenOrLessThan, this)))
        {
            return parseSignatureMember(SyntaxKind::ConstructSignature);
        }
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        auto modifiers = parseModifiers();
        if (isIndexSignature())
        {
            return parseIndexSignatureDeclaration(pos, hasJSDoc, /*decorators*/ undefined, modifiers);
        }
        return parsePropertyOrMethodSignature(pos, hasJSDoc, modifiers);
    }

    auto nextTokenIsOpenParenOrLessThan() -> boolean
    {
        nextToken();
        return token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken;
    }

    auto nextTokenIsDot()
    {
        return nextToken() == SyntaxKind::DotToken;
    }

    auto nextTokenIsOpenParenOrLessThanOrDot() -> boolean
    {
        switch (nextToken())
        {
        case SyntaxKind::OpenParenToken:
        case SyntaxKind::LessThanToken:
        case SyntaxKind::DotToken:
            return true;
        }
        return false;
    }

    auto parseTypeLiteral() -> TypeLiteralNode
    {
        auto pos = getNodePos();
        return finishNode(factory.createTypeLiteralNode(parseObjectTypeMembers()), pos);
    }

    auto parseObjectTypeMembers() -> NodeArray<TypeElement>
    {
        NodeArray<TypeElement> members;
        if (parseExpected(SyntaxKind::OpenBraceToken))
        {
            members = parseList<TypeElement>(ParsingContext::TypeMembers, std::bind(&Parser::parseTypeMember, this));
            parseExpected(SyntaxKind::CloseBraceToken);
        }
        else
        {
            members = createMissingList<TypeElement>();
        }

        return members;
    }

    auto isStartOfMappedType()
    {
        nextToken();
        if (token() == SyntaxKind::PlusToken || token() == SyntaxKind::MinusToken)
        {
            return nextToken() == SyntaxKind::ReadonlyKeyword;
        }
        if (token() == SyntaxKind::ReadonlyKeyword)
        {
            nextToken();
        }
        return token() == SyntaxKind::OpenBracketToken && nextTokenIsIdentifier() && nextToken() == SyntaxKind::InKeyword;
    }

    auto parseMappedTypeParameter()
    {
        auto pos = getNodePos();
        auto name = parseIdentifierName();
        parseExpected(SyntaxKind::InKeyword);
        auto type = parseType();
        return finishNode(factory.createTypeParameterDeclaration(name, type, /*defaultType*/ undefined), pos);
    }

    auto parseMappedType()
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::OpenBraceToken);
        Node readonlyToken;
        if (token() == SyntaxKind::ReadonlyKeyword || token() == SyntaxKind::PlusToken || token() == SyntaxKind::MinusToken)
        {
            // readonlyToken = parseTokenNode<ReadonlyKeyword, PlusToken, MinusToken>();
            readonlyToken = parseTokenNode<Node>();
            if (readonlyToken != SyntaxKind::ReadonlyKeyword)
            {
                parseExpected(SyntaxKind::ReadonlyKeyword);
            }
        }
        parseExpected(SyntaxKind::OpenBracketToken);
        auto typeParameter = parseMappedTypeParameter();
        auto nameType = parseOptional(SyntaxKind::AsKeyword) ? parseType() : undefined;
        parseExpected(SyntaxKind::CloseBracketToken);
        Node questionToken;
        if (token() == SyntaxKind::QuestionToken || token() == SyntaxKind::PlusToken || token() == SyntaxKind::MinusToken)
        {
            // questionToken = parseTokenNode<QuestionToken, PlusToken, MinusToken>();
            questionToken = parseTokenNode<Node>();
            if (questionToken != SyntaxKind::QuestionToken)
            {
                parseExpected(SyntaxKind::QuestionToken);
            }
        }
        auto type = parseTypeAnnotation();
        parseSemicolon();
        parseExpected(SyntaxKind::CloseBraceToken);
        return finishNode(factory.createMappedTypeNode(readonlyToken, typeParameter, nameType, questionToken, type), pos);
    }

    auto parseTupleElementType() -> TypeNode
    {
        auto pos = getNodePos();
        if (parseOptional(SyntaxKind::DotDotDotToken))
        {
            return finishNode(factory.createRestTypeNode(parseType()), pos);
        }
        auto type = parseType();
        if (isJSDocNullableType(type) && type->pos == type.as<JSDocNullableType>()->type->pos)
        {
            auto node = factory.createOptionalTypeNode(type.as<JSDocNullableType>()->type);
            setTextRange(node, type);
            node->flags = type->flags;
            return node;
        }
        return type;
    }

    auto isNextTokenColonOrQuestionColon()
    {
        return nextToken() == SyntaxKind::ColonToken || (token() == SyntaxKind::QuestionToken && nextToken() == SyntaxKind::ColonToken);
    }

    auto isTupleElementName()
    {
        if (token() == SyntaxKind::DotDotDotToken)
        {
            return scanner.tokenIsIdentifierOrKeyword(nextToken()) && isNextTokenColonOrQuestionColon();
        }
        return scanner.tokenIsIdentifierOrKeyword(token()) && isNextTokenColonOrQuestionColon();
    }

    auto parseTupleElementNameOrTupleElementType() -> Node
    {
        if (lookAhead<boolean>(std::bind(&Parser::isTupleElementName, this)))
        {
            auto pos = getNodePos();
            auto hasJSDoc = hasPrecedingJSDocComment();
            auto dotDotDotToken = parseOptionalToken(SyntaxKind::DotDotDotToken);
            auto name = parseIdentifierName();
            auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
            parseExpected(SyntaxKind::ColonToken);
            auto type = parseTupleElementType();
            auto node = factory.createNamedTupleMember(dotDotDotToken, name, questionToken, type);
            return withJSDoc(finishNode(node, pos), hasJSDoc);
        }
        return parseTupleElementType();
    }

    auto parseTupleType() -> TupleTypeNode
    {
        auto pos = getNodePos();
        return finishNode(factory.createTupleTypeNode(parseBracketedList<Node>(
                              ParsingContext::TupleElementTypes, std::bind(&Parser::parseTupleElementNameOrTupleElementType, this),
                              SyntaxKind::OpenBracketToken, SyntaxKind::CloseBracketToken)),
                          pos);
    }

    auto parseParenthesizedType() -> TypeNode
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::OpenParenToken);
        auto type = parseType();
        parseExpected(SyntaxKind::CloseParenToken);
        return finishNode(factory.createParenthesizedType(type), pos);
    }

    auto parseModifiersForConstructorType() -> NodeArray<Modifier>
    {
        ModifiersArray modifiers;
        if (token() == SyntaxKind::AbstractKeyword)
        {
            auto pos = getNodePos();
            nextToken();
            auto modifier = finishNode(factory.createToken(SyntaxKind::AbstractKeyword), pos);
            modifiers = createNodeArray<Modifier>(ModifiersArray(modifier), pos);
        }
        return modifiers;
    }

    auto parseFunctionOrConstructorType() -> TypeNode
    {
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        auto modifiers = parseModifiersForConstructorType();
        auto isConstructorType = parseOptional(SyntaxKind::NewKeyword);
        auto typeParameters = parseTypeParameters();
        auto parameters = parseParameters(SignatureFlags::Type);
        auto type = parseReturnType(SyntaxKind::EqualsGreaterThanToken, /*isType*/ false);
        auto node =
            isConstructorType
                ? factory.createConstructorTypeNode(modifiers, typeParameters, parameters, type).as<FunctionOrConstructorTypeNodeBase>()
                : factory.createFunctionTypeNode(typeParameters, parameters, type).as<FunctionOrConstructorTypeNodeBase>();
        if (!isConstructorType)
            copy(node->modifiers, modifiers);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseKeywordAndNoDot() -> TypeNode
    {
        auto node = parseTokenNode<TypeNode>();
        return token() == SyntaxKind::DotToken ? undefined : node;
    }

    auto parseLiteralTypeNode(boolean negative = false) -> LiteralTypeNode
    {
        auto pos = getNodePos();
        if (negative)
        {
            nextToken();
        }
        Node expression = token() == SyntaxKind::TrueKeyword || token() == SyntaxKind::FalseKeyword || token() == SyntaxKind::NullKeyword
                              ? parseTokenNode</*BooleanLiteral, NullLiteral*/ Node>().as<Expression>()
                              : parseLiteralLikeNode(token()).as<Expression>();
        if (negative)
        {
            expression = finishNode(factory.createPrefixUnaryExpression(SyntaxKind::MinusToken, expression), pos);
        }
        return finishNode(factory.createLiteralTypeNode(expression), pos);
    }

    auto isStartOfTypeOfImportType()
    {
        nextToken();
        return token() == SyntaxKind::ImportKeyword;
    }

    auto parseImportType() -> ImportTypeNode
    {
        sourceFlags |= NodeFlags::PossiblyContainsDynamicImport;
        auto pos = getNodePos();
        auto isTypeOf = parseOptional(SyntaxKind::TypeOfKeyword);
        parseExpected(SyntaxKind::ImportKeyword);
        parseExpected(SyntaxKind::OpenParenToken);
        auto type = parseType();
        parseExpected(SyntaxKind::CloseParenToken);
        auto qualifier = parseOptional(SyntaxKind::DotToken) ? parseEntityNameOfTypeReference() : undefined;
        auto typeArguments = parseTypeArgumentsOfTypeReference();
        return finishNode(factory.createImportTypeNode(type, qualifier, typeArguments, isTypeOf), pos);
    }

    auto nextTokenIsNumericOrBigIntLiteral()
    {
        nextToken();
        return token() == SyntaxKind::NumericLiteral || token() == SyntaxKind::BigIntLiteral;
    }

    auto parseNonArrayType() -> /*TypeNode*/ Node
    {
        switch (token())
        {
        case SyntaxKind::AnyKeyword:
        case SyntaxKind::UnknownKeyword:
        case SyntaxKind::StringKeyword:
        case SyntaxKind::NumberKeyword:
        case SyntaxKind::BigIntKeyword:
        case SyntaxKind::SymbolKeyword:
        case SyntaxKind::BooleanKeyword:
        case SyntaxKind::UndefinedKeyword:
        case SyntaxKind::NeverKeyword:
        case SyntaxKind::ObjectKeyword:
            // If these are followed by a dot, then parse these out.as<a>() dotted type reference instead.
            return tryParse<TypeNode>(std::bind(&Parser::parseKeywordAndNoDot, this)) || [&]() { return parseTypeReference(); };
        case SyntaxKind::AsteriskEqualsToken:
            // If there is '*=', treat it as * followed by postfix =
            scanner.reScanAsteriskEqualsToken();
            // falls through
        case SyntaxKind::AsteriskToken:
            return parseJSDocAllType();
        case SyntaxKind::QuestionQuestionToken:
            // If there is '??', treat it.as<prefix>()-'?' in JSDoc type.
            scanner.reScanQuestionToken();
            // falls through
        case SyntaxKind::QuestionToken:
            return parseJSDocUnknownOrNullableType();
        case SyntaxKind::FunctionKeyword:
            return parseJSDocFunctionType();
        case SyntaxKind::ExclamationToken:
            return parseJSDocNonNullableType();
        case SyntaxKind::NoSubstitutionTemplateLiteral:
        case SyntaxKind::StringLiteral:
        case SyntaxKind::NumericLiteral:
        case SyntaxKind::BigIntLiteral:
        case SyntaxKind::TrueKeyword:
        case SyntaxKind::FalseKeyword:
        case SyntaxKind::NullKeyword:
            return parseLiteralTypeNode();
        case SyntaxKind::MinusToken:
            return lookAhead<boolean>(std::bind(&Parser::nextTokenIsNumericOrBigIntLiteral, this))
                       ? parseLiteralTypeNode(/*negative*/ true).as<Node>()
                       : parseTypeReference().as<Node>();
        case SyntaxKind::VoidKeyword:
            return parseTokenNode<TypeNode>();
        case SyntaxKind::ThisKeyword: {
            auto thisKeyword = parseThisTypeNode();
            if (token() == SyntaxKind::IsKeyword && !scanner.hasPrecedingLineBreak())
            {
                return parseThisTypePredicate(thisKeyword);
            }
            else
            {
                return thisKeyword;
            }
        }
        case SyntaxKind::TypeOfKeyword:
            return lookAhead<boolean>(std::bind(&Parser::isStartOfTypeOfImportType, this)) ? parseImportType().as<Node>()
                                                                                           : parseTypeQuery().as<Node>();
        case SyntaxKind::OpenBraceToken:
            return lookAhead<boolean>(std::bind(&Parser::isStartOfMappedType, this)) ? parseMappedType() : parseTypeLiteral().as<Node>();
        case SyntaxKind::OpenBracketToken:
            return parseTupleType();
        case SyntaxKind::OpenParenToken:
            return parseParenthesizedType();
        case SyntaxKind::ImportKeyword:
            return parseImportType();
        case SyntaxKind::AssertsKeyword:
            return lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOnSameLine, this))
                       ? parseAssertsTypePredicate().as<Node>()
                       : parseTypeReference().as<Node>();
        case SyntaxKind::TemplateHead:
            return parseTemplateType();
        default:
            return parseTypeReference();
        }
    }

    auto isStartOfType(boolean inStartOfParameter = false) -> boolean
    {
        switch (token())
        {
        case SyntaxKind::AnyKeyword:
        case SyntaxKind::UnknownKeyword:
        case SyntaxKind::StringKeyword:
        case SyntaxKind::NumberKeyword:
        case SyntaxKind::BigIntKeyword:
        case SyntaxKind::BooleanKeyword:
        case SyntaxKind::ReadonlyKeyword:
        case SyntaxKind::SymbolKeyword:
        case SyntaxKind::UniqueKeyword:
        case SyntaxKind::VoidKeyword:
        case SyntaxKind::UndefinedKeyword:
        case SyntaxKind::NullKeyword:
        case SyntaxKind::ThisKeyword:
        case SyntaxKind::TypeOfKeyword:
        case SyntaxKind::NeverKeyword:
        case SyntaxKind::OpenBraceToken:
        case SyntaxKind::OpenBracketToken:
        case SyntaxKind::LessThanToken:
        case SyntaxKind::BarToken:
        case SyntaxKind::AmpersandToken:
        case SyntaxKind::NewKeyword:
        case SyntaxKind::StringLiteral:
        case SyntaxKind::NumericLiteral:
        case SyntaxKind::BigIntLiteral:
        case SyntaxKind::TrueKeyword:
        case SyntaxKind::FalseKeyword:
        case SyntaxKind::ObjectKeyword:
        case SyntaxKind::AsteriskToken:
        case SyntaxKind::QuestionToken:
        case SyntaxKind::ExclamationToken:
        case SyntaxKind::DotDotDotToken:
        case SyntaxKind::InferKeyword:
        case SyntaxKind::ImportKeyword:
        case SyntaxKind::AssertsKeyword:
        case SyntaxKind::NoSubstitutionTemplateLiteral:
        case SyntaxKind::TemplateHead:
            return true;
        case SyntaxKind::FunctionKeyword:
            return !inStartOfParameter;
        case SyntaxKind::MinusToken:
            return !inStartOfParameter && lookAhead<boolean>(std::bind(&Parser::nextTokenIsNumericOrBigIntLiteral, this));
        case SyntaxKind::OpenParenToken:
            // Only consider '(' the start of a type if followed by ')', '...', an identifier, a modifier,
            // or something that starts a type. We don't want to consider things like '(1)' a type.
            return !inStartOfParameter && lookAhead<boolean>(std::bind(&Parser::isStartOfParenthesizedOrFunctionType, this));
        default:
            return isIdentifier();
        }
    }

    auto isStartOfParenthesizedOrFunctionType() -> boolean
    {
        nextToken();
        return token() == SyntaxKind::CloseParenToken || isStartOfParameter(/*isJSDocParameter*/ false) || isStartOfType();
    }

    auto parsePostfixTypeOrHigher() -> TypeNode
    {
        auto pos = getNodePos();
        auto type = parseNonArrayType();
        while (!scanner.hasPrecedingLineBreak())
        {
            switch (token())
            {
            case SyntaxKind::ExclamationToken:
                nextToken();
                type = finishNode(factory.createJSDocNonNullableType(type), pos);
                break;
            case SyntaxKind::QuestionToken:
                // If next token is start of a type we have a conditional type
                if (lookAhead<boolean>(std::bind(&Parser::nextTokenIsStartOfType, this)))
                {
                    return type;
                }
                nextToken();
                type = finishNode(factory.createJSDocNullableType(type), pos);
                break;
            case SyntaxKind::OpenBracketToken:
                parseExpected(SyntaxKind::OpenBracketToken);
                if (isStartOfType())
                {
                    auto indexType = parseType();
                    parseExpected(SyntaxKind::CloseBracketToken);
                    type = finishNode(factory.createIndexedAccessTypeNode(type, indexType), pos);
                }
                else
                {
                    parseExpected(SyntaxKind::CloseBracketToken);
                    type = finishNode(factory.createArrayTypeNode(type), pos);
                }
                break;
            default:
                return type;
            }
        }
        return type;
    }

    auto parseTypeOperator(SyntaxKind operator_) -> TypeNode
    {
        auto pos = getNodePos();
        parseExpected(operator_);
        return finishNode(factory.createTypeOperatorNode(operator_, parseTypeOperatorOrHigher()), pos);
    }

    auto parseTypeParameterOfInferType()
    {
        auto pos = getNodePos();
        return finishNode(factory.createTypeParameterDeclaration(parseIdentifier(),
                                                                 /*constraint*/ undefined,
                                                                 /*defaultType*/ undefined),
                          pos);
    }

    auto parseInferType() -> InferTypeNode
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::InferKeyword);
        return finishNode(factory.createInferTypeNode(parseTypeParameterOfInferType()), pos);
    }

    auto parseTypeOperatorOrHigher() -> /*TypeNode*/ Node
    {
        auto _operator = token();
        switch (_operator)
        {
        case SyntaxKind::KeyOfKeyword:
        case SyntaxKind::UniqueKeyword:
        case SyntaxKind::ReadonlyKeyword:
            return parseTypeOperator(_operator);
        case SyntaxKind::InferKeyword:
            return parseInferType();
        }
        return parsePostfixTypeOrHigher();
    }

    auto parseFunctionOrConstructorTypeToError(boolean isInUnionType) -> TypeNode
    {
        // the auto type and constructor type shorthand notation
        // are not allowed directly in unions and intersections, but we'll
        // try to parse them gracefully and issue a helpful message.
        if (isStartOfFunctionTypeOrConstructorType())
        {
            auto type = parseFunctionOrConstructorType();
            DiagnosticMessage diagnostic;
            if (isFunctionTypeNode(type))
            {
                diagnostic =
                    isInUnionType
                        ? data::DiagnosticMessage(Diagnostics::Function_type_notation_must_be_parenthesized_when_used_in_a_union_type)
                        : data::DiagnosticMessage(
                              Diagnostics::Function_type_notation_must_be_parenthesized_when_used_in_an_intersection_type);
            }
            else
            {
                diagnostic =
                    isInUnionType
                        ? data::DiagnosticMessage(Diagnostics::Constructor_type_notation_must_be_parenthesized_when_used_in_a_union_type)
                        : data::DiagnosticMessage(
                              Diagnostics::Constructor_type_notation_must_be_parenthesized_when_used_in_an_intersection_type);
            }
            parseErrorAtRange(type, diagnostic);
            return type;
        }
        return undefined;
    }

    auto parseUnionOrIntersectionType(SyntaxKind operator_, std::function<TypeNode()> parseConstituentType,
                                      std::function<UnionOrIntersectionTypeNode(NodeArray<TypeNode>)> createTypeNode) -> TypeNode
    {
        auto pos = getNodePos();
        auto isUnionType = operator_ == SyntaxKind::BarToken;
        auto hasLeadingOperator = parseOptional(operator_);
        Node type = (hasLeadingOperator ? parseFunctionOrConstructorTypeToError(isUnionType) : undefined) ||
                    [&]() { return parseConstituentType(); };
        if (token() == operator_ || hasLeadingOperator)
        {
            auto types = NodeArray<TypeNode>(type);
            while (parseOptional(operator_))
            {
                types.push_back(parseFunctionOrConstructorTypeToError(isUnionType) || [&]() { return parseConstituentType(); });
            }
            type = finishNode(createTypeNode(createNodeArray(types, pos)), pos);
        }
        return type;
    }

    auto parseIntersectionTypeOrHigher() -> TypeNode
    {
        return parseUnionOrIntersectionType(SyntaxKind::AmpersandToken, std::bind(&Parser::parseTypeOperatorOrHigher, this),
                                            std::bind(&NodeFactory::createIntersectionTypeNode, factory, std::placeholders::_1));
    }

    auto parseUnionTypeOrHigher() -> TypeNode
    {
        return parseUnionOrIntersectionType(SyntaxKind::BarToken, std::bind(&Parser::parseIntersectionTypeOrHigher, this),
                                            std::bind(&NodeFactory::createUnionTypeNode, factory, std::placeholders::_1));
    }

    auto nextTokenIsNewKeyword() -> boolean
    {
        nextToken();
        return token() == SyntaxKind::NewKeyword;
    }

    auto isStartOfFunctionTypeOrConstructorType() -> boolean
    {
        if (token() == SyntaxKind::LessThanToken)
        {
            return true;
        }
        if (token() == SyntaxKind::OpenParenToken && lookAhead<boolean>(std::bind(&Parser::isUnambiguouslyStartOfFunctionType, this)))
        {
            return true;
        }
        return token() == SyntaxKind::NewKeyword ||
               token() == SyntaxKind::AbstractKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsNewKeyword, this));
    }

    auto skipParameterStart() -> boolean
    {
        if (isModifierKind(token()))
        {
            // Skip modifiers
            parseModifiers();
        }
        if (isIdentifier() || token() == SyntaxKind::ThisKeyword)
        {
            nextToken();
            return true;
        }
        if (token() == SyntaxKind::OpenBracketToken || token() == SyntaxKind::OpenBraceToken)
        {
            // Return true if we can parse an array or object binding pattern with no errors
            auto previousErrorCount = parseDiagnostics.size();
            parseIdentifierOrPattern();
            return previousErrorCount == parseDiagnostics.size();
        }
        return false;
    }

    auto isUnambiguouslyStartOfFunctionType() -> boolean
    {
        nextToken();
        if (token() == SyntaxKind::CloseParenToken || token() == SyntaxKind::DotDotDotToken)
        {
            // ( )
            // ( ...
            return true;
        }
        if (skipParameterStart())
        {
            // We successfully skipped modifiers (if any) and an identifier or binding pattern,
            // now see if we have something that indicates a parameter declaration
            if (token() == SyntaxKind::ColonToken || token() == SyntaxKind::CommaToken || token() == SyntaxKind::QuestionToken ||
                token() == SyntaxKind::EqualsToken)
            {
                // ( xxx :
                // ( xxx ,
                // ( xxx ?
                // ( xxx =
                return true;
            }
            if (token() == SyntaxKind::CloseParenToken)
            {
                nextToken();
                if (token() == SyntaxKind::EqualsGreaterThanToken)
                {
                    // ( xxx ) =>
                    return true;
                }
            }
        }
        return false;
    }

    auto parseTypeOrTypePredicate() -> TypeNode
    {
        auto pos = getNodePos();
        auto typePredicateVariable = isIdentifier() ? tryParse<Identifier>(std::bind(&Parser::parseTypePredicatePrefix, this)) : undefined;
        auto type = parseType();
        if (!!typePredicateVariable)
        {
            return finishNode(factory.createTypePredicateNode(/*assertsModifier*/ undefined, typePredicateVariable, type), pos);
        }
        else
        {
            return type;
        }
    }

    auto parseTypePredicatePrefix() -> Identifier
    {
        auto id = parseIdentifier();
        if (token() == SyntaxKind::IsKeyword && !scanner.hasPrecedingLineBreak())
        {
            nextToken();
            return id;
        }

        return undefined;
    }

    auto parseAssertsTypePredicate() -> TypeNode
    {
        auto pos = getNodePos();
        auto assertsModifier = parseExpectedToken(SyntaxKind::AssertsKeyword);
        auto parameterName = token() == SyntaxKind::ThisKeyword ? parseThisTypeNode().as<Node>() : parseIdentifier().as<Node>();
        auto type = parseOptional(SyntaxKind::IsKeyword) ? parseType() : undefined;
        return finishNode(factory.createTypePredicateNode(assertsModifier, parameterName, type), pos);
    }

    auto parseType() -> TypeNode
    {
        // The rules about 'yield' only apply to actual code/expression contexts.  They don't
        // apply to 'type' contexts.  So we disable these parameters here before moving on.
        return doOutsideOfContext<TypeNode>(NodeFlags::TypeExcludesFlags, std::bind(&Parser::parseTypeWorker0, this));
    }

    auto parseTypeWorker0() -> TypeNode
    {
        return parseTypeWorker();
    }

    auto parseTypeWorker(boolean noConditionalTypes = false) -> TypeNode
    {
        if (isStartOfFunctionTypeOrConstructorType())
        {
            return parseFunctionOrConstructorType();
        }
        auto pos = getNodePos();
        auto type = parseUnionTypeOrHigher();
        if (!noConditionalTypes && !scanner.hasPrecedingLineBreak() && parseOptional(SyntaxKind::ExtendsKeyword))
        {
            // The type following 'extends' is not permitted to be another conditional type
            auto extendsType = parseTypeWorker(/*noConditionalTypes*/ true);
            parseExpected(SyntaxKind::QuestionToken);
            auto trueType = parseTypeWorker();
            parseExpected(SyntaxKind::ColonToken);
            auto falseType = parseTypeWorker();
            return finishNode(factory.createConditionalTypeNode(type, extendsType, trueType, falseType), pos);
        }
        return type;
    }

    auto parseTypeAnnotation() -> TypeNode
    {
        return parseOptional(SyntaxKind::ColonToken) ? parseType() : undefined;
    }

    // EXPRESSIONS
    auto isStartOfLeftHandSideExpression() -> boolean
    {
        switch (token())
        {
        case SyntaxKind::ThisKeyword:
        case SyntaxKind::SuperKeyword:
        case SyntaxKind::NullKeyword:
        case SyntaxKind::TrueKeyword:
        case SyntaxKind::FalseKeyword:
        case SyntaxKind::NumericLiteral:
        case SyntaxKind::BigIntLiteral:
        case SyntaxKind::StringLiteral:
        case SyntaxKind::NoSubstitutionTemplateLiteral:
        case SyntaxKind::TemplateHead:
        case SyntaxKind::OpenParenToken:
        case SyntaxKind::OpenBracketToken:
        case SyntaxKind::OpenBraceToken:
        case SyntaxKind::FunctionKeyword:
        case SyntaxKind::ClassKeyword:
        case SyntaxKind::NewKeyword:
        case SyntaxKind::SlashToken:
        case SyntaxKind::SlashEqualsToken:
        case SyntaxKind::Identifier:
            return true;
        case SyntaxKind::ImportKeyword:
            return lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParenOrLessThanOrDot, this));
        default:
            return isIdentifier();
        }
    }

    auto isStartOfExpression() -> boolean
    {
        if (isStartOfLeftHandSideExpression())
        {
            return true;
        }

        switch (token())
        {
        case SyntaxKind::PlusToken:
        case SyntaxKind::MinusToken:
        case SyntaxKind::TildeToken:
        case SyntaxKind::ExclamationToken:
        case SyntaxKind::DeleteKeyword:
        case SyntaxKind::TypeOfKeyword:
        case SyntaxKind::VoidKeyword:
        case SyntaxKind::PlusPlusToken:
        case SyntaxKind::MinusMinusToken:
        case SyntaxKind::LessThanToken:
        case SyntaxKind::AwaitKeyword:
        case SyntaxKind::YieldKeyword:
        case SyntaxKind::PrivateIdentifier:
            // Yield/await always starts an expression.  Either it is an identifier (in which case
            // it is definitely an expression).  Or it's a keyword (either because we're in
            // a generator or async function, or in strict mode (or both)) and it started a yield or await expression.
            return true;
        default:
            // Error tolerance.  If we see the start of some binary operator, we consider
            // that the start of an expression.  That way we'll parse out a missing identifier,
            // give a good message about an identifier being missing, and then consume the
            // rest of the binary expression.
            if (isBinaryOperator())
            {
                return true;
            }

            return isIdentifier();
        }
    }

    auto isStartOfExpressionStatement() -> boolean
    {
        // As per the grammar, none of '{' or 'function' or 'class' can start an expression statement.
        return token() != SyntaxKind::OpenBraceToken && token() != SyntaxKind::FunctionKeyword && token() != SyntaxKind::ClassKeyword &&
               token() != SyntaxKind::AtToken && isStartOfExpression();
    }

    auto parseExpression() -> Expression
    {
        // Expression[in]:
        //      AssignmentExpression[in]
        //      Expression[in] , AssignmentExpression[in]

        // clear the decorator context when parsing Expression,.as<it>() should be unambiguous when parsing a decorator
        auto saveDecoratorContext = inDecoratorContext();
        if (saveDecoratorContext)
        {
            setDecoratorContext(/*val*/ false);
        }

        auto pos = getNodePos();
        auto expr = parseAssignmentExpressionOrHigher();
        BinaryOperatorToken operatorToken;
        while ((operatorToken = parseOptionalToken(SyntaxKind::CommaToken)))
        {
            expr = makeBinaryExpression(expr, operatorToken, parseAssignmentExpressionOrHigher(), pos);
        }

        if (saveDecoratorContext)
        {
            setDecoratorContext(/*val*/ true);
        }
        return expr;
    }

    auto parseInitializer() -> Expression
    {
        return parseOptional(SyntaxKind::EqualsToken) ? parseAssignmentExpressionOrHigher() : undefined;
    }

    auto parseAssignmentExpressionOrHigher() -> Expression
    {
        //  AssignmentExpression[in,yield]:
        //      1) ConditionalExpression[?in,?yield]
        //      2) LeftHandSideExpression = AssignmentExpression[?in,?yield]
        //      3) LeftHandSideExpression AssignmentOperator AssignmentExpression[?in,?yield]
        //      4) ArrowFunctionExpression[?in,?yield]
        //      5) AsyncArrowFunctionExpression[in,yield,await]
        //      6) [+Yield] YieldExpression[?In]
        //
        // for Note ease of implementation we treat productions '2' and '3'.as<the>() same thing.
        // (i.e. they're both BinaryExpressions with an assignment operator in it).

        // First, do the simple check if we have a YieldExpression (production '6').
        if (isYieldExpression())
        {
            return parseYieldExpression();
        }

        // Then, check if we have an arrow auto (production '4' and '5') that starts with a parenthesized
        // parameter list or is an async arrow function.
        // AsyncArrowFunctionExpression:
        //      1) async[no LineTerminator here]AsyncArrowBindingIdentifier[?Yield][no LineTerminator here]=>AsyncConciseBody[?In]
        //      2) CoverCallExpressionAndAsyncArrowHead[?Yield, ?Await][no LineTerminator here]=>AsyncConciseBody[?In]
        // Production (1) of AsyncArrowFunctionExpression is parsed in "tryParseAsyncSimpleArrowFunctionExpression".
        // And production (2) is parsed in "tryParseParenthesizedArrowFunctionExpression".
        //
        // If we do successfully parse arrow-function, we must *not* recurse for productions 1, 2 or 3. An ArrowFunction is
        // not a LeftHandSideExpression, nor does it start a ConditionalExpression.  So we are done
        // with AssignmentExpression if we see one.
        auto arrowExpression =
            tryParseParenthesizedArrowFunctionExpression() || [&]() { return tryParseAsyncSimpleArrowFunctionExpression(); };
        if (arrowExpression)
        {
            return arrowExpression;
        }

        // Now try to see if we're in production '1', '2' or '3'.  A conditional expression can
        // start with a LogicalOrExpression, while the assignment productions can only start with
        // LeftHandSideExpressions.
        //
        // So, first, we try to just parse out a BinaryExpression.  If we get something that is a
        // LeftHandSide or higher, then we can try to parse out the assignment expression part.
        // Otherwise, we try to parse out the conditional expression bit.  We want to allow any
        // binary expression here, so we pass in the 'lowest' precedence here so that it matches
        // and consumes anything.
        auto pos = getNodePos();
        auto expr = parseBinaryExpressionOrHigher(OperatorPrecedence::Lowest);

        // To avoid a look-ahead, we did not handle the case of an arrow auto with a single un-parenthesized
        // parameter ('x => ...') above. We handle it here by checking if the parsed expression was a single
        // identifier and the current token is an arrow.
        if (expr == SyntaxKind::Identifier && token() == SyntaxKind::EqualsGreaterThanToken)
        {
            return parseSimpleArrowFunctionExpression(pos, expr.as<Identifier>(), /*asyncModifier*/ undefined);
        }

        // Now see if we might be in cases '2' or '3'.
        // If the expression was a LHS expression, and we have an assignment operator, then
        // we're in '2' or '3'. Consume the assignment and return.
        //
        // we Note call reScanGreaterToken so that we get an appropriately merged token
        // for cases like `> > =` becoming `>>=`
        if (isLeftHandSideExpression(expr) && isAssignmentOperator(reScanGreaterToken()))
        {
            auto operatorToken = parseTokenNode<Node>();
            auto rightExpr = parseAssignmentExpressionOrHigher();
            return makeBinaryExpression(expr, operatorToken, rightExpr, pos);
        }

        // It wasn't an assignment or a lambda.  This is a conditional expression:
        return parseConditionalExpressionRest(expr, pos);
    }

    auto isYieldExpression() -> boolean
    {
        if (token() == SyntaxKind::YieldKeyword)
        {
            // If we have a 'yield' keyword, and this is a context where yield expressions are
            // allowed, then definitely parse out a yield expression.
            if (inYieldContext())
            {
                return true;
            }

            // We're in a context where 'yield expr' is not allowed.  However, if we can
            // definitely tell that the user was trying to parse a 'yield expr' and not
            // just a normal expr that start with a 'yield' identifier, then parse out
            // a 'yield expr'.  We can then report an error later that they are only
            // allowed in generator expressions.
            //
            // for example, if we see 'yield(foo)', then we'll have to treat that.as<an>()
            // invocation expression of something called 'yield'.  However, if we have
            // 'yield foo' then that is not legal.as<a>() normal expression, so we can
            // definitely recognize this.as<a>() yield expression.
            //
            // for now we just check if the next token is an identifier.  More heuristics
            // can be added here later.as<necessary>().  We just need to make sure that we
            // don't accidentally consume something legal.
            return lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOrLiteralOnSameLine, this));
        }

        return false;
    }

    auto nextTokenIsIdentifierOnSameLine()
    {
        nextToken();
        return !scanner.hasPrecedingLineBreak() && isIdentifier();
    }

    auto parseYieldExpression() -> YieldExpression
    {
        auto pos = getNodePos();

        // YieldExpression[In] :
        //      yield
        //      yield [no LineTerminator here] [Lexical goal InputElementRegExp]AssignmentExpression[?In, Yield]
        //      yield [no LineTerminator here] * [Lexical goal InputElementRegExp]AssignmentExpression[?In, Yield]
        nextToken();

        if (!scanner.hasPrecedingLineBreak() && (token() == SyntaxKind::AsteriskToken || isStartOfExpression()))
        {
            auto asteriskToken = parseOptionalToken(SyntaxKind::AsteriskToken);
            auto expression = parseAssignmentExpressionOrHigher();
            return finishNode(factory.createYieldExpression(asteriskToken, expression), pos);
        }
        else
        {
            // if the next token is not on the same line.as<yield>().  or we don't have an '*' or
            // the start of an expression, then this is just a simple "yield" expression.
            return finishNode(factory.createYieldExpression(/*asteriskToken*/ undefined, /*expression*/ undefined), pos);
        }
    }

    auto parseSimpleArrowFunctionExpression(number pos, Identifier identifier, NodeArray<Modifier> asyncModifier) -> ArrowFunction
    {
        Debug::_assert(token() == SyntaxKind::EqualsGreaterThanToken,
                       S("parseSimpleArrowFunctionExpression should only have been called if we had a =>"));
        auto parameter = factory.createParameterDeclaration(
            /*decorators*/ undefined,
            /*modifiers*/ undefined,
            /*dotDotDotToken*/ undefined, identifier,
            /*questionToken*/ undefined,
            /*type*/ undefined,
            /*initializer*/ undefined);
        finishNode(parameter, identifier->pos);

        auto parameters =
            createNodeArray<ParameterDeclaration>(NodeArray<ParameterDeclaration>({parameter}), parameter->pos, parameter->_end);
        auto equalsGreaterThanToken = parseExpectedToken(SyntaxKind::EqualsGreaterThanToken);
        auto body = parseArrowFunctionExpressionBody(/*isAsync*/ !!asyncModifier);
        auto node = factory.createArrowFunction(asyncModifier, /*typeParameters*/ undefined, parameters, /*type*/ undefined,
                                                equalsGreaterThanToken, body);
        return addJSDocComment(finishNode(node, pos));
    }

    auto tryParseParenthesizedArrowFunctionExpression() -> Expression
    {
        auto triState = isParenthesizedArrowFunctionExpression();
        if (triState == Tristate::False)
        {
            // It's definitely not a parenthesized arrow auto expression.
            return undefined;
        }

        // If we definitely have an arrow function, then we can just parse one, not requiring a
        // following => or { token. Otherwise, we *might* have an arrow function.  Try to parse
        // it out, but don't allow any ambiguity, and return 'undefined' if this could be an
        // expression instead.
        return triState == Tristate::True
                   ? parseParenthesizedArrowFunctionExpression(/*allowAmbiguity*/ true)
                   : tryParse<ArrowFunction>(std::bind(&Parser::parsePossibleParenthesizedArrowFunctionExpression, this));
    }

    //  True        -> We definitely expect a parenthesized arrow auto here.
    //  False       -> There *cannot* be a parenthesized arrow auto here.
    //  Unknown     -> There *might* be a parenthesized arrow auto here.
    //                 Speculatively look ahead to be sure, and rollback if not.
    auto isParenthesizedArrowFunctionExpression() -> Tristate
    {
        if (token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken || token() == SyntaxKind::AsyncKeyword)
        {
            return lookAhead<Tristate>(std::bind(&Parser::isParenthesizedArrowFunctionExpressionWorker, this));
        }

        if (token() == SyntaxKind::EqualsGreaterThanToken)
        {
            // ERROR RECOVERY TWEAK:
            // If we see a standalone => try to parse it.as<an>() arrow auto expression.as<that>()'s
            // likely what the user intended to write.
            return Tristate::True;
        }
        // Definitely not a parenthesized arrow function.
        return Tristate::False;
    }

    auto isParenthesizedArrowFunctionExpressionWorker() -> Tristate
    {
        if (token() == SyntaxKind::AsyncKeyword)
        {
            nextToken();
            if (scanner.hasPrecedingLineBreak())
            {
                return Tristate::False;
            }
            if (token() != SyntaxKind::OpenParenToken && token() != SyntaxKind::LessThanToken)
            {
                return Tristate::False;
            }
        }

        auto first = token();
        auto second = nextToken();

        if (first == SyntaxKind::OpenParenToken)
        {
            if (second == SyntaxKind::CloseParenToken)
            {
                // Simple cases: "() =>", "() -> ", and "() {".
                // This is an arrow auto with no parameters.
                // The last one is not actually an arrow function,
                // but this is probably what the user intended.
                auto third = nextToken();
                switch (third)
                {
                case SyntaxKind::EqualsGreaterThanToken:
                case SyntaxKind::ColonToken:
                case SyntaxKind::OpenBraceToken:
                    return Tristate::True;
                default:
                    return Tristate::False;
                }
            }

            // If encounter "([" or "({", this could be the start of a binding pattern.
            // Examples:
            //      ([ x ]) => { }
            //      ({ x }) => { }
            //      ([ x ])
            //      ({ x })
            if (second == SyntaxKind::OpenBracketToken || second == SyntaxKind::OpenBraceToken)
            {
                return Tristate::Unknown;
            }

            // Simple case: "(..."
            // This is an arrow auto with a rest parameter.
            if (second == SyntaxKind::DotDotDotToken)
            {
                return Tristate::True;
            }

            // Check for "(xxx yyy", where xxx is a modifier and yyy is an identifier. This
            // isn't actually allowed, but we want to treat it.as<a>() lambda so we can provide
            // a good error message.
            if (isModifierKind(second) && second != SyntaxKind::AsyncKeyword &&
                lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifier, this)))
            {
                return Tristate::True;
            }

            // If we had "(" followed by something that's not an identifier,
            // then this definitely doesn't look like a lambda.  "this" is not
            // valid, but we want to parse it and then give a semantic error.
            if (!isIdentifier() && second != SyntaxKind::ThisKeyword)
            {
                return Tristate::False;
            }

            switch (nextToken())
            {
            case SyntaxKind::ColonToken:
                // If we have something like "(a:", then we must have a
                // type-annotated parameter in an arrow auto expression.
                return Tristate::True;
            case SyntaxKind::QuestionToken:
                nextToken();
                // If we have "(a?:" or "(a?," or "(a?=" or "(a?)" then it is definitely a lambda.
                if (token() == SyntaxKind::ColonToken || token() == SyntaxKind::CommaToken || token() == SyntaxKind::EqualsToken ||
                    token() == SyntaxKind::CloseParenToken)
                {
                    return Tristate::True;
                }
                // Otherwise it is definitely not a lambda.
                return Tristate::False;
            case SyntaxKind::CommaToken:
            case SyntaxKind::EqualsToken:
            case SyntaxKind::CloseParenToken:
                // If we have "(a," or "(a=" or "(a)" this *could* be an arrow function
                return Tristate::Unknown;
            }
            // It is definitely not an arrow function
            return Tristate::False;
        }
        else
        {
            Debug::_assert(first == SyntaxKind::LessThanToken);

            // If we have "<" not followed by an identifier,
            // then this definitely is not an arrow function.
            if (!isIdentifier())
            {
                return Tristate::False;
            }

            // JSX overrides
            if (languageVariant == LanguageVariant::JSX)
            {
                auto isArrowFunctionInJsx = lookAhead<boolean>([&]() {
                    auto third = nextToken();
                    if (third == SyntaxKind::ExtendsKeyword)
                    {
                        auto fourth = nextToken();
                        switch (fourth)
                        {
                        case SyntaxKind::EqualsToken:
                        case SyntaxKind::GreaterThanToken:
                            return false;
                        default:
                            return true;
                        }
                    }
                    else if (third == SyntaxKind::CommaToken)
                    {
                        return true;
                    }
                    return false;
                });

                if (isArrowFunctionInJsx)
                {
                    return Tristate::True;
                }

                return Tristate::False;
            }

            // This *could* be a parenthesized arrow function.
            return Tristate::Unknown;
        }
    }

    auto parsePossibleParenthesizedArrowFunctionExpression() -> ArrowFunction
    {
        auto tokenPos = scanner.getTokenPos();
        if (std::find(notParenthesizedArrow.begin(), notParenthesizedArrow.end(), tokenPos) != notParenthesizedArrow.end())
        {
            return undefined;
        }

        auto result = parseParenthesizedArrowFunctionExpression(/*allowAmbiguity*/ false);
        if (!result)
        {
            notParenthesizedArrow.push_back(tokenPos);
        }

        return result;
    }

    auto tryParseAsyncSimpleArrowFunctionExpression() -> ArrowFunction
    {
        // We do a check here so that we won't be doing unnecessarily call to "lookAhead"
        if (token() == SyntaxKind::AsyncKeyword)
        {
            if (lookAhead<Tristate>(std::bind(&Parser::isUnParenthesizedAsyncArrowFunctionWorker, this)) == Tristate::True)
            {
                auto pos = getNodePos();
                auto asyncModifier = parseModifiersForArrowFunction();
                auto expr = parseBinaryExpressionOrHigher(OperatorPrecedence::Lowest);
                return parseSimpleArrowFunctionExpression(pos, expr.as<Identifier>(), asyncModifier);
            }
        }
        return undefined;
    }

    auto isUnParenthesizedAsyncArrowFunctionWorker() -> Tristate
    {
        // AsyncArrowFunctionExpression:
        //      1) async[no LineTerminator here]AsyncArrowBindingIdentifier[?Yield][no LineTerminator here]=>AsyncConciseBody[?In]
        //      2) CoverCallExpressionAndAsyncArrowHead[?Yield, ?Await][no LineTerminator here]=>AsyncConciseBody[?In]
        if (token() == SyntaxKind::AsyncKeyword)
        {
            nextToken();
            // If the "async" is followed by "=>" token then it is not a beginning of an async arrow-function
            // but instead a simple arrow-auto which will be parsed inside "parseAssignmentExpressionOrHigher"
            if (scanner.hasPrecedingLineBreak() || token() == SyntaxKind::EqualsGreaterThanToken)
            {
                return Tristate::False;
            }
            // Check for un-parenthesized AsyncArrowFunction
            auto expr = parseBinaryExpressionOrHigher(OperatorPrecedence::Lowest);
            if (!scanner.hasPrecedingLineBreak() && expr == SyntaxKind::Identifier && token() == SyntaxKind::EqualsGreaterThanToken)
            {
                return Tristate::True;
            }
        }

        return Tristate::False;
    }

    auto parseParenthesizedArrowFunctionExpression(boolean allowAmbiguity) -> ArrowFunction
    {
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        auto modifiers = parseModifiersForArrowFunction();
        auto isAsync = some(modifiers, isAsyncModifier) ? SignatureFlags::Await : SignatureFlags::None;
        // Arrow functions are never generators.
        //
        // If we're speculatively parsing a signature for a parenthesized arrow function, then
        // we have to have a complete parameter list.  Otherwise we might see something like
        // a => (b => c)
        // And think that "(b =>" was actually a parenthesized arrow auto with a missing
        // close paren.
        auto typeParameters = parseTypeParameters();

        NodeArray<ParameterDeclaration> parameters;
        if (!parseExpected(SyntaxKind::OpenParenToken))
        {
            if (!allowAmbiguity)
            {
                return undefined;
            }
            parameters = createMissingList<ParameterDeclaration>();
        }
        else
        {
            parameters = parseParametersWorker(isAsync);
            if (!parseExpected(SyntaxKind::CloseParenToken) && !allowAmbiguity)
            {
                return undefined;
            }
        }

        auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
        if (!!type && !allowAmbiguity && typeHasArrowFunctionBlockingParseError(type))
        {
            return undefined;
        }

        // Parsing a signature isn't enough.
        // Parenthesized arrow signatures often look like other valid expressions.
        // For instance:
        //  - "(x = 10)" is an assignment expression parsed.as<a>() signature with a default parameter value.
        //  - "(x,y)" is a comma expression parsed.as<a>() signature with two parameters.
        //  - "a ? (b) -> c" will have "(b) ->" parsed.as<a>() signature with a return type annotation.
        //  - "a ? (b) -> function() {}" will too, since function() is a valid JSDoc auto type.
        //
        // So we need just a bit of lookahead to ensure that it can only be a signature.
        auto hasJSDocFunctionType = !!type && isJSDocFunctionType(type);
        if (!allowAmbiguity && token() != SyntaxKind::EqualsGreaterThanToken &&
            (hasJSDocFunctionType || token() != SyntaxKind::OpenBraceToken))
        {
            // Returning undefined here will cause our caller to rewind to where we started from.
            return undefined;
        }

        // If we have an arrow, then try to parse the body. Even if not, try to parse if we
        // have an opening brace, just in case we're in an error state.
        auto lastToken = token();
        auto equalsGreaterThanToken = parseExpectedToken(SyntaxKind::EqualsGreaterThanToken);
        auto body = (lastToken == SyntaxKind::EqualsGreaterThanToken || lastToken == SyntaxKind::OpenBraceToken)
                        ? parseArrowFunctionExpressionBody(some(modifiers, isAsyncModifier))
                        : parseIdentifier().as<Node>();

        auto node = factory.createArrowFunction(modifiers, typeParameters, parameters, type, equalsGreaterThanToken, body);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseArrowFunctionExpressionBody(boolean isAsync) -> Node
    {
        if (token() == SyntaxKind::OpenBraceToken)
        {
            return parseFunctionBlock(isAsync ? SignatureFlags::Await : SignatureFlags::None);
        }

        if (token() != SyntaxKind::SemicolonToken && token() != SyntaxKind::FunctionKeyword && token() != SyntaxKind::ClassKeyword &&
            isStartOfStatement() && !isStartOfExpressionStatement())
        {
            // Check if we got a plain statement (i.e. no expression-statements, no function/class expressions/declarations)
            //
            // Here we try to recover from a potential error situation in the case where the
            // user meant to supply a block. For example, if the user wrote:
            //
            //  a =>
            //      auto v = 0;
            //  }
            //
            // they may be missing an open brace.  Check to see if that's the case so we can
            // try to recover better.  If we don't do this, then the next close curly we see may end
            // up preemptively closing the containing construct.
            //
            // even Note when 'IgnoreMissingOpenBrace' is passed, parseBody will still error.
            return parseFunctionBlock(SignatureFlags::IgnoreMissingOpenBrace | (isAsync ? SignatureFlags::Await : SignatureFlags::None));
        }

        auto savedTopLevel = topLevel;
        topLevel = false;
        auto node = isAsync ? doInAwaitContext<Expression>(std::bind(&Parser::parseAssignmentExpressionOrHigher, this))
                            : doOutsideOfAwaitContext<Expression>(std::bind(&Parser::parseAssignmentExpressionOrHigher, this));
        topLevel = savedTopLevel;
        return node;
    }

    auto parseConditionalExpressionRest(Expression leftOperand, number pos) -> Expression
    {
        // we Note are passed in an expression which was produced from parseBinaryExpressionOrHigher.
        auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
        if (!questionToken)
        {
            return leftOperand;
        }

        // we Note explicitly 'allowIn' in the whenTrue part of the condition expression, and
        // we do not that for the 'whenFalse' part.

        auto whenTrue =
            doOutsideOfContext<Expression>(disallowInAndDecoratorContext, std::bind(&Parser::parseAssignmentExpressionOrHigher, this));
        auto colonToken = parseExpectedToken(SyntaxKind::ColonToken);
        auto whenFalse = nodeIsPresent(colonToken)
                             ? parseAssignmentExpressionOrHigher().as<Node>()
                             : createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ false,
                                                             data::DiagnosticMessage(Diagnostics::_0_expected),
                                                             scanner.tokenToString(SyntaxKind::ColonToken));
        return finishNode(factory.createConditionalExpression(leftOperand, questionToken, whenTrue, colonToken, whenFalse), pos);
    }

    auto parseBinaryExpressionOrHigher(OperatorPrecedence precedence) -> Expression
    {
        auto pos = getNodePos();
        auto leftOperand = parseUnaryExpressionOrHigher();
        return parseBinaryExpressionRest(precedence, leftOperand, pos);
    }

    auto isInOrOfKeyword(SyntaxKind t) -> boolean
    {
        return t == SyntaxKind::InKeyword || t == SyntaxKind::OfKeyword;
    }

    auto parseBinaryExpressionRest(OperatorPrecedence precedence, Expression leftOperand, number pos) -> Expression
    {
        while (true)
        {
            // We either have a binary operator here, or we're finished.  We call
            // reScanGreaterToken so that we merge token sequences like > and = into >=

            reScanGreaterToken();
            auto newPrecedence = getBinaryOperatorPrecedence(token());

            // Check the precedence to see if we should "take" this operator
            // - For left associative operator (all operator but **), consume the operator,
            //   recursively call the auto below, and parse binaryExpression.as<a>() rightOperand
            //   of the caller if the new precedence of the operator is greater then or equal to the current precedence.
            //   For example:
            //      a - b - c;
            //            ^token; leftOperand = b. Return b to the caller.as<a>() rightOperand
            //      a * b - c
            //            ^token; leftOperand = b. Return b to the caller.as<a>() rightOperand
            //      a - b * c;
            //            ^token; leftOperand = b. Return b * c to the caller.as<a>() rightOperand
            // - For right associative operator (**), consume the operator, recursively call the function
            //   and parse binaryExpression.as<a>() rightOperand of the caller if the new precedence of
            //   the operator is strictly grater than the current precedence
            //   For example:
            //      a ** b ** c;
            //             ^^token; leftOperand = b. Return b ** c to the caller.as<a>() rightOperand
            //      a - b ** c;
            //            ^^token; leftOperand = b. Return b ** c to the caller.as<a>() rightOperand
            //      a ** b - c
            //             ^token; leftOperand = b. Return b to the caller.as<a>() rightOperand
            auto consumeCurrentOperator =
                token() == SyntaxKind::AsteriskAsteriskToken ? newPrecedence >= precedence : newPrecedence > precedence;

            if (!consumeCurrentOperator)
            {
                break;
            }

            if (token() == SyntaxKind::InKeyword && inDisallowInContext())
            {
                break;
            }

            if (token() == SyntaxKind::AsKeyword)
            {
                // Make sure we *do* perform ASI for constructs like this:
                //    var x = foo
                //    as (Bar)
                // This should be parsed.as<an>() initialized variable, followed
                // by a auto call to 'as' with the argument 'Bar'
                if (scanner.hasPrecedingLineBreak())
                {
                    break;
                }
                else
                {
                    nextToken();
                    leftOperand = makeAsExpression(leftOperand, parseType());
                }
            }
            else
            {
                auto tokenNode = parseTokenNode<Node>();
                auto binaryExpressionOrHigher = parseBinaryExpressionOrHigher(newPrecedence);
                leftOperand = makeBinaryExpression(leftOperand, tokenNode, binaryExpressionOrHigher, pos);
            }
        }

        return leftOperand;
    }

    auto isBinaryOperator() -> boolean
    {
        if (inDisallowInContext() && token() == SyntaxKind::InKeyword)
        {
            return false;
        }

        return getBinaryOperatorPrecedence(token()) > (OperatorPrecedence)0;
    }

    auto makeBinaryExpression(Expression left, BinaryOperatorToken operatorToken, Expression right, number pos) -> BinaryExpression
    {
        return finishNode(factory.createBinaryExpression(left, operatorToken, right), pos);
    }

    auto makeAsExpression(Expression left, TypeNode right) -> AsExpression
    {
        return finishNode(factory.createAsExpression(left, right), left->pos);
    }

    auto parsePrefixUnaryExpression() -> Node
    {
        auto pos = getNodePos();
        auto _operator = token();
        auto unaryExpression = nextTokenAnd<UnaryExpression>(std::bind(&Parser::parseSimpleUnaryExpression, this));
        return finishNode(factory.createPrefixUnaryExpression(_operator, unaryExpression), pos);
    }

    auto parseDeleteExpression() -> Node
    {
        auto pos = getNodePos();
        return finishNode(
            factory.createDeleteExpression(nextTokenAnd<UnaryExpression>(std::bind(&Parser::parseSimpleUnaryExpression, this))), pos);
    }

    auto parseTypeOfExpression() -> Node
    {
        auto pos = getNodePos();
        return finishNode(
            factory.createTypeOfExpression(nextTokenAnd<UnaryExpression>(std::bind(&Parser::parseSimpleUnaryExpression, this))), pos);
    }

    auto parseVoidExpression() -> Node
    {
        auto pos = getNodePos();
        return finishNode(factory.createVoidExpression(nextTokenAnd<UnaryExpression>(std::bind(&Parser::parseSimpleUnaryExpression, this))),
                          pos);
    }

    auto isAwaitExpression() -> boolean
    {
        if (token() == SyntaxKind::AwaitKeyword)
        {
            if (inAwaitContext())
            {
                return true;
            }

            // here we are using similar heuristics as 'isYieldExpression'
            return lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOrLiteralOnSameLine, this));
        }

        return false;
    }

    auto parseAwaitExpression() -> Node
    {
        auto pos = getNodePos();
        return finishNode(
            factory.createAwaitExpression(nextTokenAnd<UnaryExpression>(std::bind(&Parser::parseSimpleUnaryExpression, this))), pos);
    }

    /**
     * Parse ES7 exponential expression and await expression
     *
     * ES7 ExponentiationExpression:
     *      1) UnaryExpression[?Yield]
     *      2) UpdateExpression[?Yield] ** ExponentiationExpression[?Yield]
     *
     */
    auto parseUnaryExpressionOrHigher() -> Node
    {
        /**
         * ES7 UpdateExpression:
         *      1) LeftHandSideExpression[?Yield]
         *      2) LeftHandSideExpression[?Yield][no LineTerminator here]++
         *      3) LeftHandSideExpression[?Yield][no LineTerminator here]--
         *      4) ++UnaryExpression[?Yield]
         *      5) --UnaryExpression[?Yield]
         */
        if (isUpdateExpression())
        {
            auto pos = getNodePos();
            auto updateExpression = parseUpdateExpression();
            return token() == SyntaxKind::AsteriskAsteriskToken
                       ? parseBinaryExpressionRest(getBinaryOperatorPrecedence(token()), updateExpression, pos)
                       : updateExpression.as<Expression>();
        }

        /**
         * ES7 UnaryExpression:
         *      1) UpdateExpression[?yield]
         *      2) delete UpdateExpression[?yield]
         *      3) void UpdateExpression[?yield]
         *      4) typeof UpdateExpression[?yield]
         *      5) + UpdateExpression[?yield]
         *      6) - UpdateExpression[?yield]
         *      7) ~ UpdateExpression[?yield]
         *      8) ! UpdateExpression[?yield]
         */
        auto unaryOperator = token();
        auto simpleUnaryExpression = parseSimpleUnaryExpression();
        if (token() == SyntaxKind::AsteriskAsteriskToken)
        {
            auto safe_sourceText = safe_string(sourceText);
            auto pos = scanner.skipTrivia(safe_sourceText, simpleUnaryExpression->pos);
            auto end = simpleUnaryExpression->_end;
            if (simpleUnaryExpression == SyntaxKind::TypeAssertionExpression)
            {
                parseErrorAt(
                    pos, end,
                    data::DiagnosticMessage(
                        Diagnostics::
                            A_type_assertion_expression_is_not_allowed_in_the_left_hand_side_of_an_exponentiation_expression_Consider_enclosing_the_expression_in_parentheses));
            }
            else
            {
                parseErrorAt(
                    pos, end,
                    data::DiagnosticMessage(
                        Diagnostics::
                            An_unary_expression_with_the_0_operator_is_not_allowed_in_the_left_hand_side_of_an_exponentiation_expression_Consider_enclosing_the_expression_in_parentheses),
                    scanner.tokenToString(unaryOperator));
            }
        }
        return simpleUnaryExpression;
    }

    /**
     * Parse ES7 simple-unary expression or higher:
     *
     * ES7 UnaryExpression:
     *      1) UpdateExpression[?yield]
     *      2) delete UnaryExpression[?yield]
     *      3) void UnaryExpression[?yield]
     *      4) typeof UnaryExpression[?yield]
     *      5) + UnaryExpression[?yield]
     *      6) - UnaryExpression[?yield]
     *      7) ~ UnaryExpression[?yield]
     *      8) ! UnaryExpression[?yield]
     *      9) [+Await] await UnaryExpression[?yield]
     */
    auto parseSimpleUnaryExpression() -> UnaryExpression
    {
        switch (token())
        {
        case SyntaxKind::PlusToken:
        case SyntaxKind::MinusToken:
        case SyntaxKind::TildeToken:
        case SyntaxKind::ExclamationToken:
            return parsePrefixUnaryExpression();
        case SyntaxKind::DeleteKeyword:
            return parseDeleteExpression();
        case SyntaxKind::TypeOfKeyword:
            return parseTypeOfExpression();
        case SyntaxKind::VoidKeyword:
            return parseVoidExpression();
        case SyntaxKind::LessThanToken:
            // This is modified UnaryExpression grammar in TypeScript
            //  UnaryExpression (modified) ->
            //      < type > UnaryExpression
            return parseTypeAssertion();
        case SyntaxKind::AwaitKeyword:
            if (isAwaitExpression())
            {
                return parseAwaitExpression();
            }
            // falls through
        default:
            return parseUpdateExpression();
        }
    }

    /**
     * Check if the current token can possibly be an ES7 increment expression.
     *
     * ES7 UpdateExpression:
     *      LeftHandSideExpression[?Yield]
     *      LeftHandSideExpression[?Yield][no LineTerminator here]++
     *      LeftHandSideExpression[?Yield][no LineTerminator here]--
     *      ++LeftHandSideExpression[?Yield]
     *      --LeftHandSideExpression[?Yield]
     */
    auto isUpdateExpression() -> boolean
    {
        // This auto is called inside parseUnaryExpression to decide
        // whether to call parseSimpleUnaryExpression or call parseUpdateExpression directly
        switch (token())
        {
        case SyntaxKind::PlusToken:
        case SyntaxKind::MinusToken:
        case SyntaxKind::TildeToken:
        case SyntaxKind::ExclamationToken:
        case SyntaxKind::DeleteKeyword:
        case SyntaxKind::TypeOfKeyword:
        case SyntaxKind::VoidKeyword:
        case SyntaxKind::AwaitKeyword:
            return false;
        case SyntaxKind::LessThanToken:
            // If we are not in JSX context, we are parsing TypeAssertion which is an UnaryExpression
            if (languageVariant != LanguageVariant::JSX)
            {
                return false;
            }
            // We are in JSX context and the token is part of JSXElement.
            // falls through
        default:
            return true;
        }
    }

    /**
     * Parse ES7 UpdateExpression. UpdateExpression is used instead of ES6's PostFixExpression.
     *
     * ES7 UpdateExpression[yield]:
     *      1) LeftHandSideExpression[?yield]
     *      2) LeftHandSideExpression[?yield] [[no LineTerminator here]]++
     *      3) LeftHandSideExpression[?yield] [[no LineTerminator here]]--
     *      4) ++LeftHandSideExpression[?yield]
     *      5) --LeftHandSideExpression[?yield]
     * In TypeScript (2), (3) are parsed.as<PostfixUnaryExpression>(). (4), (5) are parsed.as<PrefixUnaryExpression>()
     */
    auto parseUpdateExpression() -> UpdateExpression
    {
        if (token() == SyntaxKind::PlusPlusToken || token() == SyntaxKind::MinusMinusToken)
        {
            auto pos = getNodePos();
            auto _operator = token();
            auto leftHandSideExpressionOrHigher =
                nextTokenAnd<LeftHandSideExpression>(std::bind(&Parser::parseLeftHandSideExpressionOrHigher, this));
            return finishNode(factory.createPrefixUnaryExpression(_operator, leftHandSideExpressionOrHigher), pos);
        }
        else if (languageVariant == LanguageVariant::JSX && token() == SyntaxKind::LessThanToken &&
                 lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOrGreaterThan, this)))
        {
            // JSXElement is part of primaryExpression
            return parseJsxElementOrSelfClosingElementOrFragment(/*inExpressionContext*/ true);
        }

        auto expression = parseLeftHandSideExpressionOrHigher();

        Debug::_assert(isLeftHandSideExpression(expression));
        if ((token() == SyntaxKind::PlusPlusToken || token() == SyntaxKind::MinusMinusToken) && !scanner.hasPrecedingLineBreak())
        {
            auto _operator = token();
            nextToken();
            return finishNode(factory.createPostfixUnaryExpression(expression, _operator), expression->pos);
        }

        return expression;
    }

    auto parseLeftHandSideExpressionOrHigher() -> LeftHandSideExpression
    {
        // Original Ecma:
        // See LeftHandSideExpression 11.2
        //      NewExpression
        //      CallExpression
        //
        // Our simplification:
        //
        // See LeftHandSideExpression 11.2
        //      MemberExpression
        //      CallExpression
        //
        // See comment in parseMemberExpressionOrHigher on how we replaced NewExpression with
        // MemberExpression to make our lives easier.
        //
        // to best understand the below code, it's important to see how CallExpression expands
        // out into its own productions:
        //
        // CallExpression:
        //      MemberExpression Arguments
        //      CallExpression Arguments
        //      CallExpression[Expression]
        //      CallExpression.IdentifierName
        //      import (AssignmentExpression)
        //      super Arguments
        //      super.IdentifierName
        //
        // Because of the recursion in these calls, we need to bottom out first. There are three
        // bottom out states we can run 1 into) We see 'super' which must start either of
        // the last two CallExpression productions. 2) We see 'import' which must start import call.
        // 3)we have a MemberExpression which either completes the LeftHandSideExpression,
        // or starts the beginning of the first four CallExpression productions.
        auto pos = getNodePos();
        MemberExpression expression;
        if (token() == SyntaxKind::ImportKeyword)
        {
            if (lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParenOrLessThan, this)))
            {
                // We don't want to eagerly consume all import keyword.as<import>() call expression so we look ahead to find "("
                // For example:
                //      var foo3 = require("subfolder
                //      import *.as<foo1>() from "module-from-node
                // We want this import to be a statement rather than import call expression
                sourceFlags |= NodeFlags::PossiblyContainsDynamicImport;
                expression = parseTokenNode<PrimaryExpression>();
            }
            else if (lookAhead<boolean>(std::bind(&Parser::nextTokenIsDot, this)))
            {
                // This is an 'import.*' metaproperty (i.e. 'import.meta')
                nextToken(); // advance past the 'import'
                nextToken(); // advance past the dot
                expression = finishNode(factory.createMetaProperty(SyntaxKind::ImportKeyword, parseIdentifierName()), pos);
                sourceFlags |= NodeFlags::PossiblyContainsImportMeta;
            }
            else
            {
                expression = parseMemberExpressionOrHigher();
            }
        }
        else
        {
            expression = token() == SyntaxKind::SuperKeyword ? parseSuperExpression() : parseMemberExpressionOrHigher();
        }

        // Now, we *may* be complete.  However, we might have consumed the start of a
        // CallExpression or OptionalExpression.  As such, we need to consume the rest
        // of it here to be complete.
        return parseCallExpressionRest(pos, expression);
    }

    auto parseMemberExpressionOrHigher() -> MemberExpression
    {
        // to Note make our lives simpler, we decompose the NewExpression productions and
        // place ObjectCreationExpression and FunctionExpression into PrimaryExpression.
        // like so:
        //
        //   PrimaryExpression : See 11.1
        //      this
        //      Identifier
        //      Literal
        //      ArrayLiteral
        //      ObjectLiteral
        //      (Expression)
        //      FunctionExpression
        //      new MemberExpression Arguments?
        //
        //   MemberExpression : See 11.2
        //      PrimaryExpression
        //      MemberExpression[Expression]
        //      MemberExpression.IdentifierName
        //
        //   CallExpression : See 11.2
        //      MemberExpression
        //      CallExpression Arguments
        //      CallExpression[Expression]
        //      CallExpression.IdentifierName
        //
        // Technically this is ambiguous.  i.e. CallExpression defines:
        //
        //   CallExpression:
        //      CallExpression Arguments
        //
        // If you see: "new Foo()"
        //
        // Then that could be treated.as<a>() single ObjectCreationExpression, or it could be
        // treated.as<the>() invocation of "new Foo".  We disambiguate that in code (to match
        // the original grammar) by making sure that if we see an ObjectCreationExpression
        // we always consume arguments if they are there. So we treat "new Foo()".as<an>()
        // object creation only, and not at all.as<an>() invocation.  Another way to think
        // about this is that for every "new" that we see, we will consume an argument list if
        // it is there.as<part>() of the *associated* object creation node->  Any additional
        // argument lists we see, will become invocation expressions.
        //
        // Because there are no other places in the grammar now that refer to FunctionExpression
        // or ObjectCreationExpression, it is safe to push down into the PrimaryExpression
        // production.
        //
        // Because CallExpression and MemberExpression are left recursive, we need to bottom out
        // of the recursion immediately.  So we parse out a primary expression to start with.
        auto pos = getNodePos();
        auto expression = parsePrimaryExpression();
        return parseMemberExpressionRest(pos, expression, /*allowOptionalChain*/ true);
    }

    auto parseSuperExpression() -> MemberExpression
    {
        auto pos = getNodePos();
        auto expression = parseTokenNode<PrimaryExpression>();
        if (token() == SyntaxKind::LessThanToken)
        {
            auto startPos = getNodePos();
            auto typeArguments = tryParse<NodeArray<TypeNode>>(std::bind(&Parser::parseTypeArgumentsInExpression, this));
            if (typeArguments != undefined)
            {
                parseErrorAt(startPos, getNodePos(), data::DiagnosticMessage(Diagnostics::super_may_not_use_type_arguments));
            }
        }

        if (token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::DotToken || token() == SyntaxKind::OpenBracketToken)
        {
            return expression;
        }

        // If we have seen "super" it must be followed by '(' or '.'.
        // If it wasn't then just try to parse out a '.' and report an error.
        parseExpectedToken(SyntaxKind::DotToken,
                           data::DiagnosticMessage(Diagnostics::super_must_be_followed_by_an_argument_list_or_member_access));
        // private names will never work with `super` (`super.#foo`), but that's a semantic error, not syntactic
        return finishNode(factory.createPropertyAccessExpression(
                              expression, parseRightSideOfDot(/*allowIdentifierNames*/ true, /*allowPrivateIdentifiers*/ true)),
                          pos);
    }

    auto parseJsxElementOrSelfClosingElementOrFragment(boolean inExpressionContext, number topInvalidNodePosition = -1) -> Node
    {
        auto pos = getNodePos();
        auto opening = parseJsxOpeningOrSelfClosingElementOrOpeningFragment(inExpressionContext);
        Node result;
        if (opening == SyntaxKind::JsxOpeningElement)
        {
            auto children = parseJsxChildren(opening);
            auto closingElement = parseJsxClosingElement(inExpressionContext);

            if (!tagNamesAreEquivalent(opening.as<JsxOpeningElement>()->tagName.as<JsxTagNameExpression>(),
                                       closingElement->tagName.as<JsxTagNameExpression>()))
            {
                parseErrorAtRange(closingElement, data::DiagnosticMessage(Diagnostics::Expected_corresponding_JSX_closing_tag_for_0),
                                  getTextOfNodeFromSourceText(sourceText, opening.as<JsxOpeningElement>()->tagName));
            }

            result = finishNode(factory.createJsxElement(opening, children, closingElement), pos);
        }
        else if (opening == SyntaxKind::JsxOpeningFragment)
        {
            auto jsxChildren = parseJsxChildren(opening);
            auto jsxClosingFragment = parseJsxClosingFragment(inExpressionContext);
            result = finishNode(factory.createJsxFragment(opening, jsxChildren, jsxClosingFragment), pos);
        }
        else
        {
            Debug::_assert(opening == SyntaxKind::JsxSelfClosingElement);
            // Nothing else to do for self-closing elements
            result = opening;
        }

        // If the user writes the invalid code '<div></div><div></div>' in an expression context (i.e. not wrapped in
        // an enclosing tag), we'll naively try to parse   ^ this.as<a>() 'less than' operator and the remainder of the tag
        //.as<garbage>(), which will cause the formatter to badly mangle the JSX. Perform a speculative parse of a JSX
        // element if we see a < token so that we can wrap it in a synthetic binary expression so the formatter
        // does less damage and we can report a better error.
        // Since JSX elements are invalid < operands anyway, this lookahead parse will only occur in error scenarios
        // of one sort or another.
        if (inExpressionContext && token() == SyntaxKind::LessThanToken)
        {
            auto topBadPos = topInvalidNodePosition == -1 ? result->pos : topInvalidNodePosition;
            auto invalidElement =
                tryParse<Node>([&]() { return parseJsxElementOrSelfClosingElementOrFragment(/*inExpressionContext*/ true, topBadPos); });
            if (invalidElement)
            {
                auto operatorToken = createMissingNode(SyntaxKind::CommaToken, /*reportAtCurrentPosition*/ false);
                setTextRangePosWidth(operatorToken, invalidElement->pos, 0);
                auto safe_str = safe_string(sourceText);
                parseErrorAt(scanner.skipTrivia(safe_str, topBadPos), invalidElement->_end,
                             data::DiagnosticMessage(Diagnostics::JSX_expressions_must_have_one_parent_element));
                return finishNode(factory.createBinaryExpression(result, operatorToken, invalidElement), pos);
            }
        }

        return result;
    }

    auto parseJsxText() -> JsxText
    {
        auto pos = getNodePos();
        auto tokenValue = scanner.getTokenValue();
        auto containsOnlyTriviaWhiteSpaces = currentToken == SyntaxKind::JsxTextAllWhiteSpaces;
        auto node = factory.createJsxText(tokenValue, containsOnlyTriviaWhiteSpaces);
        currentToken = scanner.scanJsxToken();
        return finishNode(node, pos);
    }

    auto parseJsxChild(Node openingTag, SyntaxKind token) -> JsxChild
    {
        switch (token)
        {
        case SyntaxKind::EndOfFileToken:
            // If we hit EOF, issue the error at the tag that lacks the closing element
            // rather than at the end of the file (which is useless)
            if (isJsxOpeningFragment(openingTag))
            {
                parseErrorAtRange(openingTag, data::DiagnosticMessage(Diagnostics::JSX_fragment_has_no_corresponding_closing_tag));
            }
            else
            {
                // We want the error span to cover only 'Foo.Bar' in < Foo.Bar >
                // or to cover only 'Foo' in < Foo >
                auto tag = openingTag.as<JsxOpeningElement>()->tagName;
                auto safe_str = safe_string(sourceText);
                auto start = scanner.skipTrivia(safe_str, tag->pos);
                parseErrorAt(start, tag->_end, data::DiagnosticMessage(Diagnostics::JSX_element_0_has_no_corresponding_closing_tag),
                             getTextOfNodeFromSourceText(sourceText, openingTag.as<JsxOpeningElement>()->tagName));
            }
            return undefined;
        case SyntaxKind::LessThanSlashToken:
        case SyntaxKind::ConflictMarkerTrivia:
            return undefined;
        case SyntaxKind::JsxText:
        case SyntaxKind::JsxTextAllWhiteSpaces:
            return parseJsxText();
        case SyntaxKind::OpenBraceToken:
            return parseJsxExpression(/*inExpressionContext*/ false);
        case SyntaxKind::LessThanToken:
            return parseJsxElementOrSelfClosingElementOrFragment(/*inExpressionContext*/ false);
        default:
            return Debug::_assertNever(/*token*/ Node());
        }
    }

    auto parseJsxChildren(Node openingTag) -> NodeArray<JsxChild>
    {
        NodeArray<JsxChild> list;
        auto listPos = getNodePos();
        auto saveParsingContext = parsingContext;
        parsingContext |= (ParsingContext)(1 << (number)ParsingContext::JsxChildren);

        while (true)
        {
            auto child = parseJsxChild(openingTag, currentToken = scanner.reScanJsxToken());
            if (!child)
                break;
            list.push_back(child);
        }

        parsingContext = saveParsingContext;
        return createNodeArray(list, listPos);
    }

    auto parseJsxAttributes() -> JsxAttributes
    {
        auto pos = getNodePos();
        return finishNode(
            factory.createJsxAttributes(parseList<Node>(ParsingContext::JsxAttributes, std::bind(&Parser::parseJsxAttribute, this))), pos);
    }

    auto parseJsxOpeningOrSelfClosingElementOrOpeningFragment(boolean inExpressionContext) -> Node
    {
        auto pos = getNodePos();

        parseExpected(SyntaxKind::LessThanToken);

        if (token() == SyntaxKind::GreaterThanToken)
        {
            // See below for explanation of scanJsxText
            scanJsxText();
            return finishNode(factory.createJsxOpeningFragment(), pos);
        }

        auto tagName = parseJsxElementName();
        auto typeArguments = (contextFlags & NodeFlags::JavaScriptFile) == NodeFlags::None ? tryParseTypeArguments() : undefined;
        auto attributes = parseJsxAttributes();

        JsxOpeningLikeElement node;

        if (token() == SyntaxKind::GreaterThanToken)
        {
            // Closing tag, so scan the immediately-following text with the JSX scanning instead
            // of regular scanning to avoid treating illegal characters (e.g. '#').as<immediate>()
            // scanning errors
            scanJsxText();
            node = factory.createJsxOpeningElement(tagName, typeArguments, attributes);
        }
        else
        {
            parseExpected(SyntaxKind::SlashToken);
            if (inExpressionContext)
            {
                parseExpected(SyntaxKind::GreaterThanToken);
            }
            else
            {
                parseExpected(SyntaxKind::GreaterThanToken, /*diagnostic*/ undefined, /*shouldAdvance*/ false);
                scanJsxText();
            }
            node = factory.createJsxSelfClosingElement(tagName, typeArguments, attributes);
        }

        return finishNode(node, pos);
    }

    auto parseJsxElementName() -> JsxTagNameExpression
    {
        auto pos = getNodePos();
        scanJsxIdentifier();
        // JsxElement can have name in the form of
        //      propertyAccessExpression
        //      primaryExpression in the form of an identifier and "this" keyword
        // We can't just simply use parseLeftHandSideExpressionOrHigher because then we will start consider class,auto etc.as<a>() keyword
        // We only want to consider "this".as<a>() primaryExpression
        JsxTagNameExpression expression =
            token() == SyntaxKind::ThisKeyword ? parseTokenNode<ThisExpression>() : parseIdentifierName().as<Node>();
        while (parseOptional(SyntaxKind::DotToken))
        {
            expression = finishNode(factory.createPropertyAccessExpression(
                                        expression, parseRightSideOfDot(/*allowIdentifierNames*/ true, /*allowPrivateIdentifiers*/ false)),
                                    pos)
                             .as<JsxTagNamePropertyAccess>();
        }
        return expression;
    }

    auto parseJsxExpression(boolean inExpressionContext) -> JsxExpression
    {
        auto pos = getNodePos();
        if (!parseExpected(SyntaxKind::OpenBraceToken))
        {
            return undefined;
        }

        Node dotDotDotToken;
        Expression expression;
        if (token() != SyntaxKind::CloseBraceToken)
        {
            dotDotDotToken = parseOptionalToken(SyntaxKind::DotDotDotToken);
            // Only an AssignmentExpression is valid here per the JSX spec,
            // but we can unambiguously parse a comma sequence and provide
            // a better error message in grammar checking.
            expression = parseExpression();
        }
        if (inExpressionContext)
        {
            parseExpected(SyntaxKind::CloseBraceToken);
        }
        else
        {
            if (parseExpected(SyntaxKind::CloseBraceToken, /*message*/ undefined, /*shouldAdvance*/ false))
            {
                scanJsxText();
            }
        }

        return finishNode(factory.createJsxExpression(dotDotDotToken, expression), pos);
    }

    auto parseJsxAttribute() -> Node
    {
        if (token() == SyntaxKind::OpenBraceToken)
        {
            return parseJsxSpreadAttribute();
        }

        scanJsxIdentifier();
        auto pos = getNodePos();
        auto identifierName = parseIdentifierName();
        auto initializer = token() != SyntaxKind::EqualsToken ? undefined
                           : scanJsxAttributeValue() == SyntaxKind::StringLiteral
                               ? parseLiteralNode().as<Node>()
                               : parseJsxExpression(/*inExpressionContext*/ true).as<Node>();
        return finishNode(factory.createJsxAttribute(identifierName, initializer), pos);
    }

    auto parseJsxSpreadAttribute() -> JsxSpreadAttribute
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::OpenBraceToken);
        parseExpected(SyntaxKind::DotDotDotToken);
        auto expression = parseExpression();
        parseExpected(SyntaxKind::CloseBraceToken);
        return finishNode(factory.createJsxSpreadAttribute(expression), pos);
    }

    auto parseJsxClosingElement(boolean inExpressionContext) -> JsxClosingElement
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::LessThanSlashToken);
        auto tagName = parseJsxElementName();
        if (inExpressionContext)
        {
            parseExpected(SyntaxKind::GreaterThanToken);
        }
        else
        {
            parseExpected(SyntaxKind::GreaterThanToken, /*diagnostic*/ undefined, /*shouldAdvance*/ false);
            scanJsxText();
        }
        return finishNode(factory.createJsxClosingElement(tagName), pos);
    }

    auto parseJsxClosingFragment(boolean inExpressionContext) -> JsxClosingFragment
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::LessThanSlashToken);
        if (scanner.tokenIsIdentifierOrKeyword(token()))
        {
            parseErrorAtRange(parseJsxElementName(),
                              data::DiagnosticMessage(Diagnostics::Expected_corresponding_closing_tag_for_JSX_fragment));
        }
        if (inExpressionContext)
        {
            parseExpected(SyntaxKind::GreaterThanToken);
        }
        else
        {
            parseExpected(SyntaxKind::GreaterThanToken, /*diagnostic*/ undefined, /*shouldAdvance*/ false);
            scanJsxText();
        }
        return finishNode(factory.createJsxJsxClosingFragment(), pos);
    }

    auto parseTypeAssertion() -> TypeAssertion
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::LessThanToken);
        auto type = parseType();
        parseExpected(SyntaxKind::GreaterThanToken);
        auto expression = parseSimpleUnaryExpression();
        return finishNode(factory.createTypeAssertion(type, expression), pos);
    }

    auto nextTokenIsIdentifierOrKeywordOrOpenBracketOrTemplate() -> boolean
    {
        nextToken();
        return scanner.tokenIsIdentifierOrKeyword(token()) || token() == SyntaxKind::OpenBracketToken || isTemplateStartOfTaggedTemplate();
    }

    auto isStartOfOptionalPropertyOrElementAccessChain() -> boolean
    {
        return token() == SyntaxKind::QuestionDotToken &&
               lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOrOpenBracketOrTemplate, this));
    }

    auto tryReparseOptionalChain(Expression node) -> boolean
    {
        if (!!(node->flags & NodeFlags::OptionalChain))
        {
            return true;
        }
        // check for an optional chain in a non-null expression
        if (isNonNullExpression(node))
        {
            auto expr = node.as<NonNullExpression>()->expression;
            while (isNonNullExpression(expr) && !(expr->flags & NodeFlags::OptionalChain))
            {
                expr = expr.as<NonNullExpression>()->expression;
            }
            if (!!(expr->flags & NodeFlags::OptionalChain))
            {
                // this is part of an optional chain. Walk down from `node` to `expression` and set the flag.
                while (isNonNullExpression(node))
                {
                    (node.asMutable<NonNullExpression>())->flags |= NodeFlags::OptionalChain;
                    node = node.as<NonNullExpression>()->expression;
                }
                return true;
            }
        }
        return false;
    }

    auto parsePropertyAccessExpressionRest(number pos, LeftHandSideExpression expression, QuestionDotToken questionDotToken)
    {
        auto name = parseRightSideOfDot(/*allowIdentifierNames*/ true, /*allowPrivateIdentifiers*/ true);
        auto isOptionalChain = (number)questionDotToken || tryReparseOptionalChain(expression);
        auto propertyAccess = isOptionalChain
                                  ? factory.createPropertyAccessChain(expression, questionDotToken, name).as<PropertyAccessExpression>()
                                  : factory.createPropertyAccessExpression(expression, name);
        if (isOptionalChain && isPrivateIdentifier(propertyAccess->name))
        {
            parseErrorAtRange(propertyAccess->name,
                              data::DiagnosticMessage(Diagnostics::An_optional_chain_cannot_contain_private_identifiers));
        }
        return finishNode(propertyAccess, pos);
    }

    auto parseElementAccessExpressionRest(number pos, LeftHandSideExpression expression, QuestionDotToken questionDotToken)
    {
        Expression argumentExpression;
        if (token() == SyntaxKind::CloseBracketToken)
        {
            argumentExpression =
                createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ true,
                                              data::DiagnosticMessage(Diagnostics::An_element_access_expression_should_take_an_argument));
        }
        else
        {
            auto argument = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
            if (isStringOrNumericLiteralLike(argument))
            {
                argument.as<LiteralLikeNode>()->text = internIdentifier(argument.as<LiteralLikeNode>()->text);
            }
            argumentExpression = argument;
        }

        parseExpected(SyntaxKind::CloseBracketToken);

        auto indexedAccess =
            (number)questionDotToken || tryReparseOptionalChain(expression)
                ? factory.createElementAccessChain(expression, questionDotToken, argumentExpression).as<ElementAccessExpression>()
                : factory.createElementAccessExpression(expression, argumentExpression);
        return finishNode(indexedAccess, pos);
    }

    auto parseMemberExpressionRest(number pos, LeftHandSideExpression expression, boolean allowOptionalChain) -> MemberExpression
    {
        while (true)
        {
            Node questionDotToken;
            auto isPropertyAccess = false;
            if (allowOptionalChain && isStartOfOptionalPropertyOrElementAccessChain())
            {
                questionDotToken = parseExpectedToken(SyntaxKind::QuestionDotToken);
                isPropertyAccess = scanner.tokenIsIdentifierOrKeyword(token());
            }
            else
            {
                isPropertyAccess = parseOptional(SyntaxKind::DotToken);
            }

            if (isPropertyAccess)
            {
                expression = parsePropertyAccessExpressionRest(pos, expression, questionDotToken);
                continue;
            }

            if (!questionDotToken && token() == SyntaxKind::ExclamationToken && !scanner.hasPrecedingLineBreak())
            {
                nextToken();
                expression = finishNode(factory.createNonNullExpression(expression), pos);
                continue;
            }

            // when in the [Decorator] context, we do not parse ElementAccess.as<it>() could be part of a ComputedPropertyName
            if ((questionDotToken || !inDecoratorContext()) && parseOptional(SyntaxKind::OpenBracketToken))
            {
                expression = parseElementAccessExpressionRest(pos, expression, questionDotToken);
                continue;
            }

            if (isTemplateStartOfTaggedTemplate())
            {
                expression = parseTaggedTemplateRest(pos, expression, questionDotToken, /*typeArguments*/ undefined);
                continue;
            }

            return expression.as<MemberExpression>();
        }
    }

    auto isTemplateStartOfTaggedTemplate() -> boolean
    {
        return token() == SyntaxKind::NoSubstitutionTemplateLiteral || token() == SyntaxKind::TemplateHead;
    }

    auto toNoSubstitutionTemplateLiteral(LiteralExpression literalExpression) -> NoSubstitutionTemplateLiteral
    {
        auto node = factory.createBaseNode<NoSubstitutionTemplateLiteral>(SyntaxKind::NoSubstitutionTemplateLiteral);
        setTextRange(node, literalExpression);
        node->rawText = literalExpression->text;
        return node;
    }

    auto parseTaggedTemplateRest(number pos, LeftHandSideExpression tag, QuestionDotToken questionDotToken,
                                 NodeArray<TypeNode> typeArguments) -> Node
    {
        auto tagExpression =
            factory.createTaggedTemplateExpression(tag, typeArguments,
                                                   token() == SyntaxKind::NoSubstitutionTemplateLiteral
                                                       ? (reScanTemplateHeadOrNoSubstitutionTemplate(),
                                                          toNoSubstitutionTemplateLiteral(parseLiteralNode()).as<TemplateLiteralLikeNode>())
                                                       : parseTemplateExpression(/*isTaggedTemplate*/ true).as<TemplateLiteralLikeNode>());
        if ((number)questionDotToken || !!(tag->flags & NodeFlags::OptionalChain))
        {
            (tagExpression.asMutable<Node>())->flags |= NodeFlags::OptionalChain;
        }
        tagExpression->questionDotToken = questionDotToken;
        return finishNode(tagExpression, pos);
    }

    auto parseCallExpressionRest(number pos, LeftHandSideExpression expression) -> LeftHandSideExpression
    {
        while (true)
        {
            expression = parseMemberExpressionRest(pos, expression, /*allowOptionalChain*/ true);
            auto questionDotToken = parseOptionalToken(SyntaxKind::QuestionDotToken);
            // handle 'foo<<T>()'
            // parse template arguments only in TypeScript files (not in JavaScript files).
            if ((contextFlags & NodeFlags::JavaScriptFile) == NodeFlags::None &&
                (token() == SyntaxKind::LessThanToken || token() == SyntaxKind::LessThanLessThanToken))
            {
                // See if this is the start of a generic invocation.  If so, consume it and
                // keep checking for postfix expressions.  Otherwise, it's just a '<' that's
                // part of an arithmetic expression.  Break out so we consume it higher in the
                // stack.
                auto typeArguments = tryParse<NodeArray<TypeNode>>(std::bind(&Parser::parseTypeArgumentsInExpression, this));
                if (!!typeArguments)
                {
                    if (isTemplateStartOfTaggedTemplate())
                    {
                        expression = parseTaggedTemplateRest(pos, expression, questionDotToken, typeArguments);
                        continue;
                    }

                    auto argumentList = parseArgumentList();
                    auto callExpr =
                        questionDotToken || tryReparseOptionalChain(expression)
                            ? factory.createCallChain(expression, questionDotToken, typeArguments, argumentList).as<CallExpression>()
                            : factory.createCallExpression(expression, typeArguments, argumentList);
                    expression = finishNode(callExpr, pos);
                    continue;
                }
            }
            else if (token() == SyntaxKind::OpenParenToken)
            {
                auto argumentList = parseArgumentList();
                auto callExpr = questionDotToken || tryReparseOptionalChain(expression)
                                    ? factory.createCallChain(expression, questionDotToken, /*typeArguments*/ undefined, argumentList)
                                          .as<CallExpression>()
                                    : factory.createCallExpression(expression, /*typeArguments*/ undefined, argumentList);
                expression = finishNode(callExpr, pos);
                continue;
            }
            if (questionDotToken)
            {
                // We failed to parse anything, so report a missing identifier here.
                auto name = createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ false,
                                                          data::DiagnosticMessage(Diagnostics::Identifier_expected));
                expression = finishNode(factory.createPropertyAccessChain(expression, questionDotToken, name), pos);
            }
            break;
        }
        return expression;
    }

    auto parseArgumentList() -> NodeArray<Expression>
    {
        parseExpected(SyntaxKind::OpenParenToken);
        auto result =
            parseDelimitedList<Expression>(ParsingContext::ArgumentExpressions, std::bind(&Parser::parseArgumentExpression, this));
        parseExpected(SyntaxKind::CloseParenToken);
        return result;
    }

    auto parseTypeArgumentsInExpression() -> NodeArray<TypeNode>
    {
        if ((contextFlags & NodeFlags::JavaScriptFile) != NodeFlags::None)
        {
            // TypeArguments must not be parsed in JavaScript files to avoid ambiguity with binary operators.
            return undefined;
        }

        if (reScanLessThanToken() != SyntaxKind::LessThanToken)
        {
            return undefined;
        }
        nextToken();

        auto typeArguments = parseDelimitedList<TypeNode>(ParsingContext::TypeArguments, std::bind(&Parser::parseType, this));
        if (!parseExpected(SyntaxKind::GreaterThanToken))
        {
            // If it doesn't have the closing `>` then it's definitely not an type argument list.
            return undefined;
        }

        // If we have a '<', then only parse this.as<a>() argument list if the type arguments
        // are complete and we have an open paren.  if we don't, rewind and return nothing.
        return !!typeArguments && canFollowTypeArgumentsInExpression() ? typeArguments : undefined;
    }

    auto canFollowTypeArgumentsInExpression() -> boolean
    {
        switch (token())
        {
        case SyntaxKind::OpenParenToken:                // foo<x>(
        case SyntaxKind::NoSubstitutionTemplateLiteral: // foo<T> `...`
        case SyntaxKind::TemplateHead:                  // foo<T> `...${100}...`
        // these are the only tokens can legally follow a type argument
        // list. So we definitely want to treat them.as<type>() arg lists.
        // falls through
        case SyntaxKind::DotToken:                     // foo<x>.
        case SyntaxKind::CloseParenToken:              // foo<x>)
        case SyntaxKind::CloseBracketToken:            // foo<x>]
        case SyntaxKind::ColonToken:                   // foo<x>:
        case SyntaxKind::SemicolonToken:               // foo<x>;
        case SyntaxKind::QuestionToken:                // foo<x>?
        case SyntaxKind::EqualsEqualsToken:            // foo<x> ==
        case SyntaxKind::EqualsEqualsEqualsToken:      // foo<x> ==
        case SyntaxKind::ExclamationEqualsToken:       // foo<x> !=
        case SyntaxKind::ExclamationEqualsEqualsToken: // foo<x> !=
        case SyntaxKind::AmpersandAmpersandToken:      // foo<x> &&
        case SyntaxKind::BarBarToken:                  // foo<x> ||
        case SyntaxKind::QuestionQuestionToken:        // foo<x> ??
        case SyntaxKind::CaretToken:                   // foo<x> ^
        case SyntaxKind::AmpersandToken:               // foo<x> &
        case SyntaxKind::BarToken:                     // foo<x> |
        case SyntaxKind::CloseBraceToken:              // foo<x> }
        case SyntaxKind::EndOfFileToken:               // foo<x>
            // these cases can't legally follow a type arg list.  However, they're not legal
            // expressions either.  The user is probably in the middle of a generic type. So
            // treat it.as<such>().
            return true;

        case SyntaxKind::CommaToken:     // foo<x>,
        case SyntaxKind::OpenBraceToken: // foo<x> {
        // We don't want to treat these.as<type>() arguments.  Otherwise we'll parse this
        //.as<an>() invocation expression.  Instead, we want to parse out the expression
        // in isolation from the type arguments.
        // falls through
        default:
            // Anything else treat.as<an>() expression.
            return false;
        }
    }

    auto parsePrimaryExpression() -> PrimaryExpression
    {
        switch (token())
        {
        case SyntaxKind::NumericLiteral:
        case SyntaxKind::BigIntLiteral:
        case SyntaxKind::StringLiteral:
        case SyntaxKind::NoSubstitutionTemplateLiteral:
            return parseLiteralNode();
        case SyntaxKind::ThisKeyword:
        case SyntaxKind::SuperKeyword:
        case SyntaxKind::NullKeyword:
        case SyntaxKind::TrueKeyword:
        case SyntaxKind::FalseKeyword:
            return parseTokenNode<PrimaryExpression>();
        case SyntaxKind::OpenParenToken:
            return parseParenthesizedExpression();
        case SyntaxKind::OpenBracketToken:
            return parseArrayLiteralExpression();
        case SyntaxKind::OpenBraceToken:
            return parseObjectLiteralExpression();
        case SyntaxKind::AsyncKeyword:
            // Async arrow functions are parsed earlier in parseAssignmentExpressionOrHigher.
            // If we encounter `async [no LineTerminator here] function` then this is an async
            // function; otherwise, its an identifier.
            if (!lookAhead<boolean>(std::bind(&Parser::nextTokenIsFunctionKeywordOnSameLine, this)))
            {
                break;
            }

            return parseFunctionExpression();
        case SyntaxKind::ClassKeyword:
            return parseClassExpression();
        case SyntaxKind::FunctionKeyword:
            return parseFunctionExpression();
        case SyntaxKind::NewKeyword:
            return parseNewExpressionOrNewDotTarget();
        case SyntaxKind::SlashToken:
        case SyntaxKind::SlashEqualsToken:
            if (reScanSlashToken() == SyntaxKind::RegularExpressionLiteral)
            {
                return parseLiteralNode();
            }
            break;
        case SyntaxKind::TemplateHead:
            return parseTemplateExpression(/* isTaggedTemplate */ false);
        }

        return parseIdentifier(data::DiagnosticMessage(Diagnostics::Expression_expected));
    }

    auto parseParenthesizedExpression() -> ParenthesizedExpression
    {
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        parseExpected(SyntaxKind::OpenParenToken);
        auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        parseExpected(SyntaxKind::CloseParenToken);
        return withJSDoc(finishNode(factory.createParenthesizedExpression(expression), pos), hasJSDoc);
    }

    auto parseSpreadElement() -> Expression
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::DotDotDotToken);
        auto expression = parseAssignmentExpressionOrHigher();
        return finishNode(factory.createSpreadElement(expression), pos);
    }

    auto parseArgumentOrArrayLiteralElement() -> Expression
    {
        return token() == SyntaxKind::DotDotDotToken ? parseSpreadElement()
               : token() == SyntaxKind::CommaToken   ? finishNode(factory.createOmittedExpression(), getNodePos()).as<Expression>()
                                                     : parseAssignmentExpressionOrHigher();
    }

    auto parseArgumentExpression() -> Expression
    {
        return doOutsideOfContext<Expression>(disallowInAndDecoratorContext, std::bind(&Parser::parseArgumentOrArrayLiteralElement, this));
    }

    auto parseArrayLiteralExpression() -> Node
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::OpenBracketToken);
        auto multiLine = scanner.hasPrecedingLineBreak();
        auto elements = parseDelimitedList<Expression>(ParsingContext::ArrayLiteralMembers,
                                                       std::bind(&Parser::parseArgumentOrArrayLiteralElement, this));
        parseExpected(SyntaxKind::CloseBracketToken);
        return finishNode(factory.createArrayLiteralExpression(elements, multiLine), pos);
    }

    auto parseObjectLiteralElement() -> ObjectLiteralElementLike
    {
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();

        if (parseOptionalToken(SyntaxKind::DotDotDotToken))
        {
            auto expression = parseAssignmentExpressionOrHigher();
            return withJSDoc(finishNode(factory.createSpreadAssignment(expression), pos), hasJSDoc);
        }

        auto decorators = parseDecorators();
        auto modifiers = parseModifiers();

        if (parseContextualModifier(SyntaxKind::GetKeyword))
        {
            return parseAccessorDeclaration(pos, hasJSDoc, decorators, modifiers, SyntaxKind::GetAccessor);
        }
        if (parseContextualModifier(SyntaxKind::SetKeyword))
        {
            return parseAccessorDeclaration(pos, hasJSDoc, decorators, modifiers, SyntaxKind::SetAccessor);
        }

        auto asteriskToken = parseOptionalToken(SyntaxKind::AsteriskToken);
        auto tokenIsIdentifier = isIdentifier();
        auto name = parsePropertyName();

        // Disallowing of optional property assignments and definite assignment assertion happens in the grammar checker.
        auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
        auto exclamationToken = parseOptionalToken(SyntaxKind::ExclamationToken);

        if (asteriskToken || token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken)
        {
            return parseMethodDeclaration(pos, hasJSDoc, decorators, modifiers, asteriskToken, name, questionToken, exclamationToken);
        }

        // check if it is short-hand property assignment or normal property assignment
        // if NOTE token is EqualsToken it is interpreted.as<CoverInitializedName>() production
        // CoverInitializedName[Yield] :
        //     IdentifierReference[?Yield] Initializer[In, ?Yield]
        // this is necessary because ObjectLiteral productions are also used to cover grammar for ObjectAssignmentPattern
        PropertyAssignment node;
        auto isShorthandPropertyAssignment = tokenIsIdentifier && (token() != SyntaxKind::ColonToken);
        if (isShorthandPropertyAssignment)
        {
            auto equalsToken = parseOptionalToken(SyntaxKind::EqualsToken);
            auto objectAssignmentInitializer =
                equalsToken ? allowInAnd<Expression>(std::bind(&Parser::parseAssignmentExpressionOrHigher, this)) : undefined;
            auto nodeSpa = factory.createShorthandPropertyAssignment(name.as<Identifier>(), objectAssignmentInitializer);
            // Save equals token for error reporting.
            // TODO(rbuckton) -> Consider manufacturing this when we need to report an error.as<it>() is otherwise not useful.
            nodeSpa->equalsToken = equalsToken;
            node = nodeSpa.as<PropertyAssignment>();
        }
        else
        {
            parseExpected(SyntaxKind::ColonToken);
            auto initializer = allowInAnd<Expression>(std::bind(&Parser::parseAssignmentExpressionOrHigher, this));
            node = factory.createPropertyAssignment(name, initializer);
        }
        // Decorators, Modifiers, questionToken, and exclamationToken are not supported by property assignments and are reported in the
        // grammar checker
        node->decorators = decorators;
        copy(node->modifiers, modifiers);
        node->questionToken = questionToken;
        node->exclamationToken = exclamationToken;
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseObjectLiteralExpression() -> Node
    {
        auto pos = getNodePos();
        auto openBracePosition = scanner.getTokenPos();
        parseExpected(SyntaxKind::OpenBraceToken);
        auto multiLine = scanner.hasPrecedingLineBreak();
        auto properties = parseDelimitedList<ObjectLiteralElementLike>(ParsingContext::ObjectLiteralMembers,
                                                                       std::bind(&Parser::parseObjectLiteralElement, this),
                                                                       /*considerSemicolonAsDelimiter*/ true);
        if (!parseExpected(SyntaxKind::CloseBraceToken))
        {
            auto lastError = lastOrUndefined(parseDiagnostics);
            if (!!lastError && lastError->code == data::DiagnosticMessage(Diagnostics::_0_expected).code)
            {
                addRelatedInfo(lastError, createDetachedDiagnostic(
                                              fileName, openBracePosition, 1,
                                              data::DiagnosticMessage(Diagnostics::The_parser_expected_to_find_a_to_match_the_token_here)));
            }
        }
        return finishNode(factory.createObjectLiteralExpression(properties, multiLine), pos);
    }

    auto parseFunctionExpression() -> FunctionExpression
    {
        // GeneratorExpression:
        //      function* BindingIdentifier [Yield][opt](FormalParameters[Yield]){ GeneratorBody }
        //
        // FunctionExpression:
        //      auto BindingIdentifier[opt](FormalParameters){ FunctionBody }
        auto saveDecoratorContext = inDecoratorContext();
        if (saveDecoratorContext)
        {
            setDecoratorContext(/*val*/ false);
        }

        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        auto modifiers = parseModifiers();
        parseExpected(SyntaxKind::FunctionKeyword);
        auto asteriskToken = parseOptionalToken(SyntaxKind::AsteriskToken);
        auto isGenerator = asteriskToken ? SignatureFlags::Yield : SignatureFlags::None;
        auto isAsync = some(modifiers, isAsyncModifier) ? SignatureFlags::Await : SignatureFlags::None;
        auto name = !!isGenerator && !!isAsync
                        ? doInYieldAndAwaitContext<Identifier>(std::bind(&Parser::parseOptionalBindingIdentifier, this))
                    : !!isGenerator ? doInYieldContext<Identifier>(std::bind(&Parser::parseOptionalBindingIdentifier, this))
                    : !!isAsync     ? doInAwaitContext<Identifier>(std::bind(&Parser::parseOptionalBindingIdentifier, this))
                                    : parseOptionalBindingIdentifier();

        auto typeParameters = parseTypeParameters();
        auto parameters = parseParameters(isGenerator | isAsync);
        auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
        auto body = parseFunctionBlock(isGenerator | isAsync);

        if (saveDecoratorContext)
        {
            setDecoratorContext(/*val*/ true);
        }

        auto node = factory.createFunctionExpression(modifiers, asteriskToken, name, typeParameters, parameters, type, body);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseOptionalBindingIdentifier() -> Identifier
    {
        return isBindingIdentifier() ? parseBindingIdentifier() : undefined;
    }

    auto parseNewExpressionOrNewDotTarget() -> Node
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::NewKeyword);
        if (parseOptional(SyntaxKind::DotToken))
        {
            auto name = parseIdentifierName();
            return finishNode(factory.createMetaProperty(SyntaxKind::NewKeyword, name), pos);
        }

        auto expressionPos = getNodePos();
        auto expression = parsePrimaryExpression();
        NodeArray<TypeNode> typeArguments;
        while (true)
        {
            expression = parseMemberExpressionRest(expressionPos, expression, /*allowOptionalChain*/ false);
            typeArguments = tryParse<NodeArray<TypeNode>>(std::bind(&Parser::parseTypeArgumentsInExpression, this));
            if (isTemplateStartOfTaggedTemplate())
            {
                Debug::_assert(
                    !!typeArguments,
                    S("Expected a type argument list; all plain tagged template starts should be consumed in 'parseMemberExpressionRest'"));
                expression = parseTaggedTemplateRest(expressionPos, expression, /*optionalChain*/ undefined, typeArguments);
                typeArguments = undefined;
            }
            break;
        }

        NodeArray<Expression> argumentsArray;
        if (token() == SyntaxKind::OpenParenToken)
        {
            argumentsArray = parseArgumentList();
        }
        else if (!!typeArguments)
        {
            parseErrorAt(pos, scanner.getStartPos(),
                         data::DiagnosticMessage(
                             Diagnostics::A_new_expression_with_type_arguments_must_always_be_followed_by_a_parenthesized_argument_list));
        }
        return finishNode(factory.createNewExpression(expression, typeArguments, argumentsArray), pos);
    }

    // STATEMENTS
    auto parseBlock(boolean ignoreMissingOpenBrace, DiagnosticMessage diagnosticMessage = undefined) -> Block
    {
        auto pos = getNodePos();
        auto openBracePosition = scanner.getTokenPos();
        if (parseExpected(SyntaxKind::OpenBraceToken, diagnosticMessage) || ignoreMissingOpenBrace)
        {
            auto multiLine = scanner.hasPrecedingLineBreak();
            auto statements = parseList<Statement>(ParsingContext::BlockStatements, std::bind(&Parser::parseStatement, this));
            if (!parseExpected(SyntaxKind::CloseBraceToken))
            {
                auto lastError = lastOrUndefined(parseDiagnostics);
                if (!!lastError && lastError->code == data::DiagnosticMessage(Diagnostics::_0_expected).code)
                {
                    addRelatedInfo(lastError,
                                   createDetachedDiagnostic(
                                       fileName, openBracePosition, 1,
                                       data::DiagnosticMessage(Diagnostics::The_parser_expected_to_find_a_to_match_the_token_here)));
                }
            }
            return finishNode(factory.createBlock(statements, multiLine), pos);
        }
        else
        {
            auto statements = createMissingList<Statement>();
            return finishNode(factory.createBlock(statements, /*multiLine*/ false), pos);
        }
    }

    auto parseFunctionBlock(SignatureFlags flags, DiagnosticMessage diagnosticMessage = undefined) -> Block
    {
        auto savedYieldContext = inYieldContext();
        setYieldContext(!!(flags & SignatureFlags::Yield));

        auto savedAwaitContext = inAwaitContext();
        setAwaitContext(!!(flags & SignatureFlags::Await));

        auto savedTopLevel = topLevel;
        topLevel = false;

        // We may be in a [Decorator] context when parsing a auto expression or
        // arrow function. The body of the auto is not in [Decorator] context->
        auto saveDecoratorContext = inDecoratorContext();
        if (saveDecoratorContext)
        {
            setDecoratorContext(/*val*/ false);
        }

        auto block = parseBlock(!!(flags & SignatureFlags::IgnoreMissingOpenBrace), diagnosticMessage);

        if (saveDecoratorContext)
        {
            setDecoratorContext(/*val*/ true);
        }

        topLevel = savedTopLevel;
        setYieldContext(savedYieldContext);
        setAwaitContext(savedAwaitContext);

        return block;
    }

    auto parseEmptyStatement() -> Statement
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::SemicolonToken);
        return finishNode(factory.createEmptyStatement(), pos);
    }

    auto parseIfStatement() -> IfStatement
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::IfKeyword);
        parseExpected(SyntaxKind::OpenParenToken);
        auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        parseExpected(SyntaxKind::CloseParenToken);
        auto thenStatement = parseStatement();
        auto elseStatement = parseOptional(SyntaxKind::ElseKeyword) ? parseStatement() : undefined;
        return finishNode(factory.createIfStatement(expression, thenStatement, elseStatement), pos);
    }

    auto parseDoStatement() -> DoStatement
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::DoKeyword);
        auto statement = parseStatement();
        parseExpected(SyntaxKind::WhileKeyword);
        parseExpected(SyntaxKind::OpenParenToken);
        auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        parseExpected(SyntaxKind::CloseParenToken);

        // https From://mail.mozilla.org/pipermail/es-discuss/2011-August/016188.html
        // 157 min --- All allen at wirfs-brock.com CONF --- "do{;}while(false)false" prohibited in
        // spec but allowed in consensus reality. Approved -- this is the de-facto standard whereby
        //  do;while(0)x will have a semicolon inserted before x.
        parseOptional(SyntaxKind::SemicolonToken);
        return finishNode(factory.createDoStatement(statement, expression), pos);
    }

    auto parseWhileStatement() -> WhileStatement
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::WhileKeyword);
        parseExpected(SyntaxKind::OpenParenToken);
        auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        parseExpected(SyntaxKind::CloseParenToken);
        auto statement = parseStatement();
        return finishNode(factory.createWhileStatement(expression, statement), pos);
    }

    auto parseForOrForInOrForOfStatement() -> Statement
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::ForKeyword);
        auto awaitToken = parseOptionalToken(SyntaxKind::AwaitKeyword);
        parseExpected(SyntaxKind::OpenParenToken);

        Node initializer;
        if (token() != SyntaxKind::SemicolonToken)
        {
            if (token() == SyntaxKind::VarKeyword || token() == SyntaxKind::LetKeyword || token() == SyntaxKind::ConstKeyword)
            {
                initializer = parseVariableDeclarationList(/*inForStatementInitializer*/ true);
            }
            else
            {
                initializer = disallowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
            }
        }

        Node node;
        if (awaitToken ? parseExpected(SyntaxKind::OfKeyword) : parseOptional(SyntaxKind::OfKeyword))
        {
            auto expression = allowInAnd<Expression>(std::bind(&Parser::parseAssignmentExpressionOrHigher, this));
            parseExpected(SyntaxKind::CloseParenToken);
            node = factory.createForOfStatement(awaitToken, initializer, expression, parseStatement());
        }
        else if (parseOptional(SyntaxKind::InKeyword))
        {
            auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
            parseExpected(SyntaxKind::CloseParenToken);
            node = factory.createForInStatement(initializer, expression, parseStatement());
        }
        else
        {
            parseExpected(SyntaxKind::SemicolonToken);
            auto condition = token() != SyntaxKind::SemicolonToken && token() != SyntaxKind::CloseParenToken
                                 ? allowInAnd<Expression>(std::bind(&Parser::parseExpression, this))
                                 : undefined;
            parseExpected(SyntaxKind::SemicolonToken);
            auto incrementor =
                token() != SyntaxKind::CloseParenToken ? allowInAnd<Expression>(std::bind(&Parser::parseExpression, this)) : undefined;
            parseExpected(SyntaxKind::CloseParenToken);
            node = factory.createForStatement(initializer, condition, incrementor, parseStatement());
        }

        return finishNode(node, pos);
    }

    auto parseBreakOrContinueStatement(SyntaxKind kind) -> BreakOrContinueStatement
    {
        auto pos = getNodePos();

        parseExpected(kind == SyntaxKind::BreakStatement ? SyntaxKind::BreakKeyword : SyntaxKind::ContinueKeyword);
        auto label = canParseSemicolon() ? undefined : parseIdentifier();

        parseSemicolon();
        auto node = kind == SyntaxKind::BreakStatement ? factory.createBreakStatement(label).as<BreakOrContinueStatement>()
                                                       : factory.createContinueStatement(label).as<BreakOrContinueStatement>();
        return finishNode(node, pos);
    }

    auto parseReturnStatement() -> ReturnStatement
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::ReturnKeyword);
        auto expression = canParseSemicolon() ? undefined : allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        parseSemicolon();
        return finishNode(factory.createReturnStatement(expression), pos);
    }

    auto parseWithStatement() -> WithStatement
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::WithKeyword);
        parseExpected(SyntaxKind::OpenParenToken);
        auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        parseExpected(SyntaxKind::CloseParenToken);
        auto statement = doInsideOfContext<Statement>(NodeFlags::InWithStatement, std::bind(&Parser::parseStatement, this));
        return finishNode(factory.createWithStatement(expression, statement), pos);
    }

    auto parseCaseClause() -> CaseClause
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::CaseKeyword);
        auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        parseExpected(SyntaxKind::ColonToken);
        auto statements = parseList<Statement>(ParsingContext::SwitchClauseStatements, std::bind(&Parser::parseStatement, this));
        return finishNode(factory.createCaseClause(expression, statements), pos);
    }

    auto parseDefaultClause() -> DefaultClause
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::DefaultKeyword);
        parseExpected(SyntaxKind::ColonToken);
        auto statements = parseList<Statement>(ParsingContext::SwitchClauseStatements, std::bind(&Parser::parseStatement, this));
        return finishNode(factory.createDefaultClause(statements), pos);
    }

    auto parseCaseOrDefaultClause() -> CaseOrDefaultClause
    {
        return token() == SyntaxKind::CaseKeyword ? (CaseOrDefaultClause)parseCaseClause() : (CaseOrDefaultClause)parseDefaultClause();
    }

    auto parseCaseBlock() -> CaseBlock
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::OpenBraceToken);
        auto clauses = parseList<CaseOrDefaultClause>(ParsingContext::SwitchClauses, std::bind(&Parser::parseCaseOrDefaultClause, this));
        parseExpected(SyntaxKind::CloseBraceToken);
        return finishNode(factory.createCaseBlock(clauses), pos);
    }

    auto parseSwitchStatement() -> SwitchStatement
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::SwitchKeyword);
        parseExpected(SyntaxKind::OpenParenToken);
        auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        parseExpected(SyntaxKind::CloseParenToken);
        auto caseBlock = parseCaseBlock();
        return finishNode(factory.createSwitchStatement(expression, caseBlock), pos);
    }

    auto parseThrowStatement() -> ThrowStatement
    {
        // ThrowStatement[Yield] :
        //      throw [no LineTerminator here]Expression[In, ?Yield];

        auto pos = getNodePos();
        parseExpected(SyntaxKind::ThrowKeyword);

        // Because of automatic semicolon insertion, we need to report error if this
        // throw could be terminated with a semicolon.  we Note can't call 'parseExpression'
        // directly.as<that>() might consume an expression on the following line.
        // Instead, we create a "missing" identifier, but don't report an error. The actual error
        // will be reported in the grammar walker.
        auto expression = scanner.hasPrecedingLineBreak() ? undefined : allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        if (expression == undefined)
        {
            identifierCount++;
            expression = finishNode(factory.createIdentifier(string()), getNodePos());
        }
        parseSemicolon();
        return finishNode(factory.createThrowStatement(expression), pos);
    }

    // Review TODO for error recovery
    auto parseTryStatement() -> TryStatement
    {
        auto pos = getNodePos();

        parseExpected(SyntaxKind::TryKeyword);
        auto tryBlock = parseBlock(/*ignoreMissingOpenBrace*/ false);
        auto catchClause = token() == SyntaxKind::CatchKeyword ? parseCatchClause() : undefined;

        // If we don't have a catch clause, then we must have a finally clause.  Try to parse
        // one out no matter what.
        Block finallyBlock;
        if (!catchClause || token() == SyntaxKind::FinallyKeyword)
        {
            parseExpected(SyntaxKind::FinallyKeyword);
            finallyBlock = parseBlock(/*ignoreMissingOpenBrace*/ false);
        }

        return finishNode(factory.createTryStatement(tryBlock, catchClause, finallyBlock), pos);
    }

    auto parseCatchClause() -> CatchClause
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::CatchKeyword);

        VariableDeclaration variableDeclaration;
        if (parseOptional(SyntaxKind::OpenParenToken))
        {
            variableDeclaration = parseVariableDeclaration();
            parseExpected(SyntaxKind::CloseParenToken);
        }
        else
        {
            // Keep shape of node to avoid degrading performance.
            variableDeclaration = undefined;
        }

        auto block = parseBlock(/*ignoreMissingOpenBrace*/ false);
        return finishNode(factory.createCatchClause(variableDeclaration, block), pos);
    }

    auto parseDebuggerStatement() -> Statement
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::DebuggerKeyword);
        parseSemicolon();
        return finishNode(factory.createDebuggerStatement(), pos);
    }

    auto parseExpressionOrLabeledStatement() -> Node
    {
        // Avoiding having to do the lookahead for a labeled statement by just trying to parse
        // out an expression, seeing if it is identifier and then seeing if it is followed by
        // a colon.
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        Node node;
        auto hasParen = token() == SyntaxKind::OpenParenToken;
        auto expression = allowInAnd<Expression>(std::bind(&Parser::parseExpression, this));
        if (ts::isIdentifier(expression) && parseOptional(SyntaxKind::ColonToken))
        {
            node = factory.createLabeledStatement(expression, parseStatement());
        }
        else
        {
            parseSemicolon();
            node = factory.createExpressionStatement(expression);
            if (hasParen)
            {
                // do not parse the same jsdoc twice
                hasJSDoc = false;
            }
        }
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto nextTokenIsIdentifierOrKeywordOnSameLine() -> boolean
    {
        nextToken();
        return scanner.tokenIsIdentifierOrKeyword(token()) && !scanner.hasPrecedingLineBreak();
    }

    auto nextTokenIsClassKeywordOnSameLine() -> boolean
    {
        nextToken();
        return token() == SyntaxKind::ClassKeyword && !scanner.hasPrecedingLineBreak();
    }

    auto nextTokenIsFunctionKeywordOnSameLine() -> boolean
    {
        nextToken();
        return token() == SyntaxKind::FunctionKeyword && !scanner.hasPrecedingLineBreak();
    }

    auto nextTokenIsIdentifierOrKeywordOrLiteralOnSameLine() -> boolean
    {
        nextToken();
        return (scanner.tokenIsIdentifierOrKeyword(token()) || token() == SyntaxKind::NumericLiteral ||
                token() == SyntaxKind::BigIntLiteral || token() == SyntaxKind::StringLiteral) &&
               !scanner.hasPrecedingLineBreak();
    }

    auto isDeclaration() -> boolean
    {
        while (true)
        {
            switch (token())
            {
            case SyntaxKind::VarKeyword:
            case SyntaxKind::LetKeyword:
            case SyntaxKind::ConstKeyword:
            case SyntaxKind::FunctionKeyword:
            case SyntaxKind::ClassKeyword:
            case SyntaxKind::EnumKeyword:
                return true;

            // 'declare', 'module', 'namespace', 'interface'* and 'type' are all legal JavaScript identifiers;
            // however, an identifier cannot be followed by another identifier on the same line. This is what we
            // count on to parse out the respective declarations. For instance, we exploit this to say that
            //
            //    namespace n
            //
            // can be none other than the beginning of a namespace declaration, but need to respect that JavaScript sees
            //
            //    namespace
            //    n
            //
            //.as<the>() identifier 'namespace' on one line followed by the identifier 'n' on another.
            // We need to look one token ahead to see if it permissible to try parsing a declaration.
            //
            // *Note*: 'interface' is actually a strict mode reserved word. So while
            //
            //   "use strict"
            //   interface
            //   I {}
            //
            // could be legal, it would add complexity for very little gain.
            case SyntaxKind::InterfaceKeyword:
            case SyntaxKind::TypeKeyword:
                return nextTokenIsIdentifierOnSameLine();
            case SyntaxKind::ModuleKeyword:
            case SyntaxKind::NamespaceKeyword:
                return nextTokenIsIdentifierOrStringLiteralOnSameLine();
            case SyntaxKind::AbstractKeyword:
            case SyntaxKind::AsyncKeyword:
            case SyntaxKind::DeclareKeyword:
            case SyntaxKind::PrivateKeyword:
            case SyntaxKind::ProtectedKeyword:
            case SyntaxKind::PublicKeyword:
            case SyntaxKind::ReadonlyKeyword:
                nextToken();
                // ASI takes effect for this modifier.
                if (scanner.hasPrecedingLineBreak())
                {
                    return false;
                }
                continue;

            case SyntaxKind::GlobalKeyword:
                nextToken();
                return token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::Identifier || token() == SyntaxKind::ExportKeyword;

            case SyntaxKind::ImportKeyword:
                nextToken();
                return token() == SyntaxKind::StringLiteral || token() == SyntaxKind::AsteriskToken ||
                       token() == SyntaxKind::OpenBraceToken || scanner.tokenIsIdentifierOrKeyword(token());
            case SyntaxKind::ExportKeyword: {
                auto currentToken = nextToken();
                if (currentToken == SyntaxKind::TypeKeyword)
                {
                    currentToken = lookAhead<SyntaxKind>(std::bind(&Parser::nextToken, this));
                }
                if (currentToken == SyntaxKind::EqualsToken || currentToken == SyntaxKind::AsteriskToken ||
                    currentToken == SyntaxKind::OpenBraceToken || currentToken == SyntaxKind::DefaultKeyword ||
                    currentToken == SyntaxKind::AsKeyword)
                {
                    return true;
                }
                continue;
            }
            case SyntaxKind::StaticKeyword:
                nextToken();
                continue;
            default:
                return false;
            }
        }
    }

    auto isStartOfDeclaration() -> boolean
    {
        return lookAhead<boolean>(std::bind(&Parser::isDeclaration, this));
    }

    auto isStartOfStatement() -> boolean
    {
        switch (token())
        {
        case SyntaxKind::AtToken:
        case SyntaxKind::SemicolonToken:
        case SyntaxKind::OpenBraceToken:
        case SyntaxKind::VarKeyword:
        case SyntaxKind::LetKeyword:
        case SyntaxKind::FunctionKeyword:
        case SyntaxKind::ClassKeyword:
        case SyntaxKind::EnumKeyword:
        case SyntaxKind::IfKeyword:
        case SyntaxKind::DoKeyword:
        case SyntaxKind::WhileKeyword:
        case SyntaxKind::ForKeyword:
        case SyntaxKind::ContinueKeyword:
        case SyntaxKind::BreakKeyword:
        case SyntaxKind::ReturnKeyword:
        case SyntaxKind::WithKeyword:
        case SyntaxKind::SwitchKeyword:
        case SyntaxKind::ThrowKeyword:
        case SyntaxKind::TryKeyword:
        case SyntaxKind::DebuggerKeyword:
        // 'catch' and 'finally' do not actually indicate that the code is part of a statement,
        // however, we say they are here so that we may gracefully parse them and error later.
        // falls through
        case SyntaxKind::CatchKeyword:
        case SyntaxKind::FinallyKeyword:
            return true;

        case SyntaxKind::ImportKeyword:
            return isStartOfDeclaration() || lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParenOrLessThanOrDot, this));

        case SyntaxKind::ConstKeyword:
        case SyntaxKind::ExportKeyword:
            return isStartOfDeclaration();

        case SyntaxKind::AsyncKeyword:
        case SyntaxKind::DeclareKeyword:
        case SyntaxKind::InterfaceKeyword:
        case SyntaxKind::ModuleKeyword:
        case SyntaxKind::NamespaceKeyword:
        case SyntaxKind::TypeKeyword:
        case SyntaxKind::GlobalKeyword:
            // When these don't start a declaration, they're an identifier in an expression statement
            return true;

        case SyntaxKind::PublicKeyword:
        case SyntaxKind::PrivateKeyword:
        case SyntaxKind::ProtectedKeyword:
        case SyntaxKind::StaticKeyword:
        case SyntaxKind::ReadonlyKeyword:
            // When these don't start a declaration, they may be the start of a class member if an identifier
            // immediately follows. Otherwise they're an identifier in an expression statement.
            return isStartOfDeclaration() || !lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeywordOnSameLine, this));

        default:
            return isStartOfExpression();
        }
    }

    auto nextTokenIsIdentifierOrStartOfDestructuring()
    {
        nextToken();
        return isIdentifier() || token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::OpenBracketToken;
    }

    auto isLetDeclaration()
    {
        // In ES6 'let' always starts a lexical declaration if followed by an identifier or {
        // or [.
        return lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrStartOfDestructuring, this));
    }

    auto parseStatement() -> Statement
    {
        switch (token())
        {
        case SyntaxKind::SemicolonToken:
            return parseEmptyStatement();
        case SyntaxKind::OpenBraceToken:
            return parseBlock(/*ignoreMissingOpenBrace*/ false);
        case SyntaxKind::VarKeyword:
            return parseVariableStatement(getNodePos(), hasPrecedingJSDocComment(), /*decorators*/ undefined, /*modifiers*/ undefined);
        case SyntaxKind::LetKeyword:
            if (isLetDeclaration())
            {
                return parseVariableStatement(getNodePos(), hasPrecedingJSDocComment(), /*decorators*/ undefined, /*modifiers*/ undefined);
            }
            break;
        case SyntaxKind::FunctionKeyword:
            return parseFunctionDeclaration(getNodePos(), hasPrecedingJSDocComment(), /*decorators*/ undefined, /*modifiers*/ undefined);
        case SyntaxKind::ClassKeyword:
            return parseClassDeclaration(getNodePos(), hasPrecedingJSDocComment(), /*decorators*/ undefined, /*modifiers*/ undefined);
        case SyntaxKind::IfKeyword:
            return parseIfStatement();
        case SyntaxKind::DoKeyword:
            return parseDoStatement();
        case SyntaxKind::WhileKeyword:
            return parseWhileStatement();
        case SyntaxKind::ForKeyword:
            return parseForOrForInOrForOfStatement();
        case SyntaxKind::ContinueKeyword:
            return parseBreakOrContinueStatement(SyntaxKind::ContinueStatement);
        case SyntaxKind::BreakKeyword:
            return parseBreakOrContinueStatement(SyntaxKind::BreakStatement);
        case SyntaxKind::ReturnKeyword:
            return parseReturnStatement();
        case SyntaxKind::WithKeyword:
            return parseWithStatement();
        case SyntaxKind::SwitchKeyword:
            return parseSwitchStatement();
        case SyntaxKind::ThrowKeyword:
            return parseThrowStatement();
        case SyntaxKind::TryKeyword:
        // Include 'catch' and 'finally' for error recovery.
        // falls through
        case SyntaxKind::CatchKeyword:
        case SyntaxKind::FinallyKeyword:
            return parseTryStatement();
        case SyntaxKind::DebuggerKeyword:
            return parseDebuggerStatement();
        case SyntaxKind::AtToken:
            return parseDeclaration();
        case SyntaxKind::AsyncKeyword:
        case SyntaxKind::InterfaceKeyword:
        case SyntaxKind::TypeKeyword:
        case SyntaxKind::ModuleKeyword:
        case SyntaxKind::NamespaceKeyword:
        case SyntaxKind::DeclareKeyword:
        case SyntaxKind::ConstKeyword:
        case SyntaxKind::EnumKeyword:
        case SyntaxKind::ExportKeyword:
        case SyntaxKind::ImportKeyword:
        case SyntaxKind::PrivateKeyword:
        case SyntaxKind::ProtectedKeyword:
        case SyntaxKind::PublicKeyword:
        case SyntaxKind::AbstractKeyword:
        case SyntaxKind::StaticKeyword:
        case SyntaxKind::ReadonlyKeyword:
        case SyntaxKind::GlobalKeyword:
            if (isStartOfDeclaration())
            {
                return parseDeclaration();
            }
            break;
        }
        return parseExpressionOrLabeledStatement();
    }

    auto isDeclareModifier(Modifier modifier) -> boolean
    {
        return modifier == SyntaxKind::DeclareKeyword;
    }

    auto parseDeclaration() -> Statement
    {
        // Can TODO we hold onto the parsed decorators/modifiers and advance the scanner
        //       if we can't reuse the declaration, so that we don't do this work twice?
        //
        // `parseListElement` attempted to get the reused node at this position,
        // but the ambient context flag was not yet set, so the node appeared
        // not reusable in that context->
        auto isAmbient = some(lookAhead<NodeArray<Modifier>>([&]() {
                                  parseDecorators();
                                  return parseModifiers();
                              }),
                              std::bind(&Parser::isDeclareModifier, this, std::placeholders::_1));
        if (isAmbient)
        {
            auto node = tryReuseAmbientDeclaration();
            if (node)
            {
                return node;
            }
        }

        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        auto decorators = parseDecorators();
        auto modifiers = parseModifiers();
        if (isAmbient)
        {
            for (auto &m : modifiers)
            {
                (m.asMutable<Node>())->flags |= NodeFlags::Ambient;
            }
            return doInsideOfContext<Node>(NodeFlags::Ambient,
                                           [&]() { return parseDeclarationWorker(pos, hasJSDoc, decorators, modifiers); });
        }
        else
        {
            return parseDeclarationWorker(pos, hasJSDoc, decorators, modifiers);
        }
    }

    auto tryReuseAmbientDeclaration() -> Statement
    {
        return doInsideOfContext<Statement>(NodeFlags::Ambient, [&]() {
            auto node = currentNode(parsingContext);
            if (node)
            {
                return consumeNode(node).as<Statement>();
            }

            return (Statement)undefined;
        });
    }

    auto parseDeclarationWorker(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers) -> Statement
    {
        switch (token())
        {
        case SyntaxKind::VarKeyword:
        case SyntaxKind::LetKeyword:
        case SyntaxKind::ConstKeyword:
            return parseVariableStatement(pos, hasJSDoc, decorators, modifiers);
        case SyntaxKind::FunctionKeyword:
            return parseFunctionDeclaration(pos, hasJSDoc, decorators, modifiers);
        case SyntaxKind::ClassKeyword:
            return parseClassDeclaration(pos, hasJSDoc, decorators, modifiers);
        case SyntaxKind::InterfaceKeyword:
            return parseInterfaceDeclaration(pos, hasJSDoc, decorators, modifiers);
        case SyntaxKind::TypeKeyword:
            return parseTypeAliasDeclaration(pos, hasJSDoc, decorators, modifiers);
        case SyntaxKind::EnumKeyword:
            return parseEnumDeclaration(pos, hasJSDoc, decorators, modifiers);
        case SyntaxKind::GlobalKeyword:
        case SyntaxKind::ModuleKeyword:
        case SyntaxKind::NamespaceKeyword:
            return parseModuleDeclaration(pos, hasJSDoc, decorators, modifiers);
        case SyntaxKind::ImportKeyword:
            return parseImportDeclarationOrImportEqualsDeclaration(pos, hasJSDoc, decorators, modifiers);
        case SyntaxKind::ExportKeyword:
            nextToken();
            switch (token())
            {
            case SyntaxKind::DefaultKeyword:
            case SyntaxKind::EqualsToken:
                return parseExportAssignment(pos, hasJSDoc, decorators, modifiers);
            case SyntaxKind::AsKeyword:
                return parseNamespaceExportDeclaration(pos, hasJSDoc, decorators, modifiers);
            default:
                return parseExportDeclaration(pos, hasJSDoc, decorators, modifiers);
            }
        default:
            if (!!decorators || !!modifiers)
            {
                // We reached this point because we encountered decorators and/or modifiers and assumed a declaration
                // would follow. For recovery and error reporting purposes, return an incomplete declaration.
                auto missing = createMissingNode<MissingDeclaration>(SyntaxKind::MissingDeclaration, /*reportAtCurrentPosition*/ true,
                                                                     data::DiagnosticMessage(Diagnostics::Declaration_expected));
                setTextRangePos(missing, pos);
                missing->decorators = decorators;
                copy(missing->modifiers, modifiers);
                return missing;
            }
            return undefined; // GH TODO#18217
        }
    }

    auto nextTokenIsIdentifierOrStringLiteralOnSameLine() -> boolean
    {
        nextToken();
        return !scanner.hasPrecedingLineBreak() && (isIdentifier() || token() == SyntaxKind::StringLiteral);
    }

    auto parseFunctionBlockOrSemicolon(SignatureFlags flags, DiagnosticMessage diagnosticMessage = undefined) -> Block
    {
        if (token() != SyntaxKind::OpenBraceToken && canParseSemicolon())
        {
            parseSemicolon();
            return undefined;
        }

        return parseFunctionBlock(flags, diagnosticMessage);
    }

    // DECLARATIONS

    auto parseArrayBindingElement() -> ArrayBindingElement
    {
        auto pos = getNodePos();
        if (token() == SyntaxKind::CommaToken)
        {
            return finishNode(factory.createOmittedExpression(), pos);
        }
        auto dotDotDotToken = parseOptionalToken(SyntaxKind::DotDotDotToken);
        auto name = parseIdentifierOrPattern();
        auto initializer = parseInitializer();
        return finishNode(factory.createBindingElement(dotDotDotToken, /*propertyName*/ undefined, name, initializer), pos);
    }

    auto parseObjectBindingElement() -> BindingElement
    {
        auto pos = getNodePos();
        auto dotDotDotToken = parseOptionalToken(SyntaxKind::DotDotDotToken);
        auto tokenIsIdentifier = isBindingIdentifier();
        /*PropertyName*/ auto propertyName = parsePropertyName();
        BindingName name;
        if (tokenIsIdentifier && token() != SyntaxKind::ColonToken)
        {
            name = propertyName.as<Identifier>();
            propertyName = undefined;
        }
        else
        {
            parseExpected(SyntaxKind::ColonToken);
            name = parseIdentifierOrPattern();
        }
        auto initializer = parseInitializer();
        return finishNode(factory.createBindingElement(dotDotDotToken, propertyName, name, initializer), pos);
    }

    auto parseObjectBindingPattern() -> ObjectBindingPattern
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::OpenBraceToken);
        auto elements =
            parseDelimitedList<BindingElement>(ParsingContext::ObjectBindingElements, std::bind(&Parser::parseObjectBindingElement, this));
        parseExpected(SyntaxKind::CloseBraceToken);
        return finishNode(factory.createObjectBindingPattern(elements), pos);
    }

    auto parseArrayBindingPattern() -> ArrayBindingPattern
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::OpenBracketToken);
        auto elements = parseDelimitedList<ArrayBindingElement>(ParsingContext::ArrayBindingElements,
                                                                std::bind(&Parser::parseArrayBindingElement, this));
        parseExpected(SyntaxKind::CloseBracketToken);
        return finishNode(factory.createArrayBindingPattern(elements), pos);
    }

    auto isBindingIdentifierOrPrivateIdentifierOrPattern() -> boolean
    {
        return token() == SyntaxKind::OpenBraceToken || token() == SyntaxKind::OpenBracketToken ||
               token() == SyntaxKind::PrivateIdentifier || isBindingIdentifier();
    }

    auto parseIdentifierOrPattern(DiagnosticMessage privateIdentifierDiagnosticMessage = undefined) -> Node
    {
        if (token() == SyntaxKind::OpenBracketToken)
        {
            return parseArrayBindingPattern();
        }
        if (token() == SyntaxKind::OpenBraceToken)
        {
            return parseObjectBindingPattern();
        }
        return parseBindingIdentifier(privateIdentifierDiagnosticMessage);
    }

    auto parseVariableDeclarationAllowExclamation()
    {
        return parseVariableDeclaration(/*allowExclamation*/ true);
    }

    auto parseVariableDeclaration0() -> VariableDeclaration
    {
        return parseVariableDeclaration();
    }

    auto parseVariableDeclaration(boolean allowExclamation = false) -> VariableDeclaration
    {
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        auto name =
            parseIdentifierOrPattern(data::DiagnosticMessage(Diagnostics::Private_identifiers_are_not_allowed_in_variable_declarations));
        ExclamationToken exclamationToken;
        if (allowExclamation && name == SyntaxKind::Identifier && token() == SyntaxKind::ExclamationToken &&
            !scanner.hasPrecedingLineBreak())
        {
            exclamationToken = parseTokenNode<Token<SyntaxKind::ExclamationToken>>();
        }
        auto type = parseTypeAnnotation();
        auto initializer = isInOrOfKeyword(token()) ? undefined : parseInitializer();
        auto node = factory.createVariableDeclaration(name, exclamationToken, type, initializer);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseVariableDeclarationList(boolean inForStatementInitializer) -> VariableDeclarationList
    {
        auto pos = getNodePos();

        auto flags = NodeFlags::None;
        switch (token())
        {
        case SyntaxKind::VarKeyword:
            break;
        case SyntaxKind::LetKeyword:
            flags |= NodeFlags::Let;
            break;
        case SyntaxKind::ConstKeyword:
            flags |= NodeFlags::Const;
            break;
        default:
            Debug::fail<void>();
        }

        nextToken();

        // The user may have written the following:
        //
        //    for (auto of X) { }
        //
        // In this case, we want to parse an empty declaration list, and then parse 'of'
        //.as<a>() keyword. The reason this is not automatic is that 'of' is a valid identifier.
        // So we need to look ahead to determine if 'of' should be treated.as<a>() keyword in
        // this context->
        // The checker will then give an error that there is an empty declaration list.
        NodeArray<VariableDeclaration> declarations;
        if (token() == SyntaxKind::OfKeyword && lookAhead<boolean>(std::bind(&Parser::canFollowContextualOfKeyword, this)))
        {
            declarations = createMissingList<VariableDeclaration>();
        }
        else
        {
            auto savedDisallowIn = inDisallowInContext();
            setDisallowInContext(inForStatementInitializer);

            declarations = parseDelimitedList<VariableDeclaration>(
                ParsingContext::VariableDeclarations,
                inForStatementInitializer
                    ? (std::function<VariableDeclaration()>)std::bind(&Parser::parseVariableDeclaration0, this)
                    : (std::function<VariableDeclaration()>)std::bind(&Parser::parseVariableDeclarationAllowExclamation, this));

            setDisallowInContext(savedDisallowIn);
        }

        return finishNode(factory.createVariableDeclarationList(declarations, flags), pos);
    }

    auto canFollowContextualOfKeyword() -> boolean
    {
        return nextTokenIsIdentifier() && nextToken() == SyntaxKind::CloseParenToken;
    }

    auto parseVariableStatement(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> VariableStatement
    {
        auto declarationList = parseVariableDeclarationList(/*inForStatementInitializer*/ false);
        parseSemicolon();
        auto node = factory.createVariableStatement(modifiers, declarationList);
        // Decorators are not allowed on a variable statement, so we keep track of them to report them in the grammar checker.
        node->decorators = decorators;
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseFunctionDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> FunctionDeclaration
    {
        auto savedAwaitContext = inAwaitContext();
        auto modifierFlags = modifiersToFlags(modifiers);
        parseExpected(SyntaxKind::FunctionKeyword);
        auto asteriskToken = parseOptionalToken(SyntaxKind::AsteriskToken);
        // We don't parse the name here in await context, instead we will report a grammar error in the checker.
        auto name = !!(modifierFlags & ModifierFlags::Default) ? parseOptionalBindingIdentifier() : parseBindingIdentifier();
        auto isGenerator = asteriskToken ? SignatureFlags::Yield : SignatureFlags::None;
        auto isAsync = !!(modifierFlags & ModifierFlags::Async) ? SignatureFlags::Await : SignatureFlags::None;
        auto typeParameters = parseTypeParameters();
        if (!!(modifierFlags & ModifierFlags::Export))
            setAwaitContext(/*value*/ true);
        auto parameters = parseParameters(isGenerator | isAsync);
        auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
        auto body = parseFunctionBlockOrSemicolon(isGenerator | isAsync, data::DiagnosticMessage(Diagnostics::or_expected));
        setAwaitContext(savedAwaitContext);
        auto node = factory.createFunctionDeclaration(decorators, modifiers, asteriskToken, name, typeParameters, parameters, type, body);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseConstructorName() -> boolean
    {
        if (token() == SyntaxKind::ConstructorKeyword)
        {
            return parseExpected(SyntaxKind::ConstructorKeyword);
        }
        if (token() == SyntaxKind::StringLiteral &&
            lookAhead<SyntaxKind>(std::bind(&Parser::nextToken, this)) == SyntaxKind::OpenParenToken)
        {
            return tryParse<boolean>([&]() {
                auto literalNode = parseLiteralNode();
                return literalNode->text == S("constructor");
            });
        }

        return false;
    }

    auto tryParseConstructorDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> ConstructorDeclaration
    {
        return tryParse<ConstructorDeclaration>([&]() {
            if (parseConstructorName())
            {
                auto typeParameters = parseTypeParameters();
                auto parameters = parseParameters(SignatureFlags::None);
                auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
                auto body = parseFunctionBlockOrSemicolon(SignatureFlags::None, data::DiagnosticMessage(Diagnostics::or_expected));
                auto node = factory.createConstructorDeclaration(decorators, modifiers, parameters, body);
                // Attach `typeParameters` and `type` if they exist so that we can report them in the grammar checker.
                node->typeParameters = typeParameters;
                node->type = type;
                return withJSDoc(finishNode(node, pos), hasJSDoc);
            }

            return (Node)undefined;
        });

        return undefined;
    }

    auto parseMethodDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers,
                                AsteriskToken asteriskToken, PropertyName name, QuestionToken questionToken,
                                ExclamationToken exclamationToken, DiagnosticMessage diagnosticMessage = undefined) -> MethodDeclaration
    {
        auto isGenerator = asteriskToken == SyntaxKind::AsteriskToken ? SignatureFlags::Yield : SignatureFlags::None;
        auto isAsync = some(modifiers, isAsyncModifier) ? SignatureFlags::Await : SignatureFlags::None;
        auto typeParameters = parseTypeParameters();
        auto parameters = parseParameters(isGenerator | isAsync);
        auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
        auto body = parseFunctionBlockOrSemicolon(isGenerator | isAsync, diagnosticMessage);
        auto node = factory.createMethodDeclaration(decorators, modifiers, asteriskToken, name, questionToken, typeParameters, parameters,
                                                    type, body);
        // An exclamation token on a method is invalid syntax and will be handled by the grammar checker
        node->exclamationToken = exclamationToken;
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parsePropertyDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers,
                                  PropertyName name, QuestionToken questionToken) -> PropertyDeclaration
    {
        auto exclamationToken =
            !questionToken && !scanner.hasPrecedingLineBreak() ? parseOptionalToken(SyntaxKind::ExclamationToken) : undefined;
        auto type = parseTypeAnnotation();
        auto initializer = doOutsideOfContext<Expression>(NodeFlags::YieldContext | NodeFlags::AwaitContext | NodeFlags::DisallowInContext,
                                                          std::bind(&Parser::parseInitializer, this));
        parseSemicolon();
        auto node = factory.createPropertyDeclaration(
            decorators, modifiers, name, questionToken.as<Node>() || [&]() { return exclamationToken.as<Node>(); }, type, initializer);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parsePropertyOrMethodDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> Node
    {
        auto asteriskToken = parseOptionalToken(SyntaxKind::AsteriskToken);
        auto name = parsePropertyName();
        // this Note is not legal.as<per>() the grammar.  But we allow it in the parser and
        // report an error in the grammar checker.
        auto questionToken = parseOptionalToken(SyntaxKind::QuestionToken);
        if (asteriskToken || token() == SyntaxKind::OpenParenToken || token() == SyntaxKind::LessThanToken)
        {
            return parseMethodDeclaration(pos, hasJSDoc, decorators, modifiers, asteriskToken, name, questionToken,
                                          /*exclamationToken*/ undefined, data::DiagnosticMessage(Diagnostics::or_expected));
        }
        return parsePropertyDeclaration(pos, hasJSDoc, decorators, modifiers, name, questionToken);
    }

    auto parseAccessorDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers,
                                  SyntaxKind kind) -> AccessorDeclaration
    {
        auto name = parsePropertyName();
        auto typeParameters = parseTypeParameters();
        auto parameters = parseParameters(SignatureFlags::None);
        auto type = parseReturnType(SyntaxKind::ColonToken, /*isType*/ false);
        auto body = parseFunctionBlockOrSemicolon(SignatureFlags::None);
        auto node =
            kind == SyntaxKind::GetAccessor
                ? factory.createGetAccessorDeclaration(decorators, modifiers, name, parameters, type, body).as<AccessorDeclaration>()
                : factory.createSetAccessorDeclaration(decorators, modifiers, name, parameters, body).as<AccessorDeclaration>();
        // Keep track of `typeParameters` (for both) and `type` (for setters) if they were parsed those indicate grammar errors
        node->typeParameters = typeParameters;
        if (!!type && node == SyntaxKind::SetAccessor)
            (node.asMutable<SetAccessorDeclaration>())->type = type;
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto isClassMemberStart() -> boolean
    {
        auto idToken = SyntaxKind::Unknown;

        if (token() == SyntaxKind::AtToken)
        {
            return true;
        }

        // Eat up all modifiers, but hold on to the last one in case it is actually an identifier.
        while (isModifierKind(token()))
        {
            idToken = token();
            // If the idToken is a class modifier (protected, private, public, and static), it is
            // certain that we are starting to parse class member. This allows better error recovery
            // Example:
            //      public foo() ...     // true
            //      public @dec blah ... // true; we will then report an error later
            //      public ...    // true; we will then report an error later
            if (isClassMemberModifier(idToken))
            {
                return true;
            }

            nextToken();
        }

        if (token() == SyntaxKind::AsteriskToken)
        {
            return true;
        }

        // Try to get the first property-like token following all modifiers.
        // This can either be an identifier or the 'get' or 'set' keywords.
        if (isLiteralPropertyName())
        {
            idToken = token();
            nextToken();
        }

        // Index signatures and computed properties are class members; we can parse.
        if (token() == SyntaxKind::OpenBracketToken)
        {
            return true;
        }

        // If we were able to get any potential identifier...
        if (idToken != SyntaxKind::Unknown)
        {
            // If we have a non-keyword identifier, or if we have an accessor, then it's safe to parse.
            if (!isKeyword(idToken) || idToken == SyntaxKind::SetKeyword || idToken == SyntaxKind::GetKeyword)
            {
                return true;
            }

            // If it *is* a keyword, but not an accessor, check a little farther along
            // to see if it should actually be parsed.as<a>() class member.
            switch (token())
            {
            case SyntaxKind::OpenParenToken:   // Method declaration
            case SyntaxKind::LessThanToken:    // Generic Method declaration
            case SyntaxKind::ExclamationToken: // Non-null assertion on property name
            case SyntaxKind::ColonToken:       // Type Annotation for declaration
            case SyntaxKind::EqualsToken:      // Initializer for declaration
            case SyntaxKind::QuestionToken:    // Not valid, but permitted so that it gets caught later on.
                return true;
            default:
                // Covers
                //  - Semicolons     (declaration termination)
                //  - Closing braces (end-of-class, must be declaration)
                //  - End-of-files   (not valid, but permitted so that it gets caught later on)
                //  - Line-breaks    (enabling *automatic semicolon insertion*)
                return canParseSemicolon();
            }
        }

        return false;
    }

    auto parseDecoratorExpression()
    {
        if (inAwaitContext() && token() == SyntaxKind::AwaitKeyword)
        {
            // `@await` is is disallowed in an [Await] context, but can cause parsing to go off the rails
            // This simply parses the missing identifier and moves on.
            auto pos = getNodePos();
            auto awaitExpression = parseIdentifier(data::DiagnosticMessage(Diagnostics::Expression_expected));
            nextToken();
            auto memberExpression = parseMemberExpressionRest(pos, awaitExpression, /*allowOptionalChain*/ true);
            return parseCallExpressionRest(pos, memberExpression);
        }
        return parseLeftHandSideExpressionOrHigher();
    }

    auto tryParseDecorator() -> Decorator
    {
        auto pos = getNodePos();
        if (!parseOptional(SyntaxKind::AtToken))
        {
            return undefined;
        }
        auto expression = doInDecoratorContext<LeftHandSideExpression>(std::bind(&Parser::parseDecoratorExpression, this));
        return finishNode(factory.createDecorator(expression), pos);
    }

    auto parseDecorators() -> NodeArray<Decorator>
    {
        auto pos = getNodePos();
        NodeArray<Decorator> list;
        Decorator decorator;
        while (!!(decorator = tryParseDecorator()))
        {
            list = append(list, decorator);
        }
        return !!list ? createNodeArray(list, pos) : undefined;
    }

    auto tryParseModifier(boolean permitInvalidConstAsModifier) -> Modifier
    {
        auto pos = getNodePos();
        auto kind = token();

        if (token() == SyntaxKind::ConstKeyword && permitInvalidConstAsModifier)
        {
            // We need to ensure that any subsequent modifiers appear on the same line
            // so that when 'const' is a standalone declaration, we don't issue an error.
            if (!tryParse<boolean>(std::bind(&Parser::nextTokenIsOnSameLineAndCanFollowModifier, this)))
            {
                return undefined;
            }
        }
        else
        {
            if (!parseAnyContextualModifier())
            {
                return undefined;
            }
        }

        return finishNode(factory.createToken(kind), pos);
    }

    /*
     * There are situations in which a modifier like 'const' will appear unexpectedly, such.as<on>() a class member.
     * In those situations, if we are entirely sure that 'const' is not valid on its own (such.as<when>() ASI takes effect
     * and turns it into a standalone declaration), then it is better to parse it and report an error later.
     *
     * In such situations, 'permitInvalidConstAsModifier' should be set to true.
     */
    auto parseModifiers(boolean permitInvalidConstAsModifier = false) -> NodeArray<Modifier>
    {
        auto pos = getNodePos();
        NodeArray<Modifier> list;
        Modifier modifier;
        while (modifier = tryParseModifier(permitInvalidConstAsModifier))
        {
            list = append(list, modifier);
        }
        return !!list ? createNodeArray(list, pos) : undefined;
    }

    auto parseModifiersForArrowFunction() -> NodeArray<Modifier>
    {
        NodeArray<Modifier> modifiers;
        if (token() == SyntaxKind::AsyncKeyword)
        {
            auto pos = getNodePos();
            nextToken();
            auto modifier = finishNode(factory.createToken(SyntaxKind::AsyncKeyword), pos);
            modifiers = createNodeArray<Modifier>(NodeArray<Modifier>({modifier}), pos);
        }
        return modifiers;
    }

    auto parseClassElement() -> ClassElement
    {
        auto pos = getNodePos();
        if (token() == SyntaxKind::SemicolonToken)
        {
            nextToken();
            return finishNode(factory.createSemicolonClassElement(), pos);
        }

        auto hasJSDoc = hasPrecedingJSDocComment();
        auto decorators = parseDecorators();
        auto modifiers = parseModifiers(/*permitInvalidConstAsModifier*/ true);

        if (parseContextualModifier(SyntaxKind::GetKeyword))
        {
            return parseAccessorDeclaration(pos, hasJSDoc, decorators, modifiers, SyntaxKind::GetAccessor);
        }

        if (parseContextualModifier(SyntaxKind::SetKeyword))
        {
            return parseAccessorDeclaration(pos, hasJSDoc, decorators, modifiers, SyntaxKind::SetAccessor);
        }

        if (token() == SyntaxKind::ConstructorKeyword || token() == SyntaxKind::StringLiteral)
        {
            auto constructorDeclaration = tryParseConstructorDeclaration(pos, hasJSDoc, decorators, modifiers);
            if (!!constructorDeclaration)
            {
                return constructorDeclaration;
            }
        }

        if (isIndexSignature())
        {
            return parseIndexSignatureDeclaration(pos, hasJSDoc, decorators, modifiers);
        }

        // It is very important that we check this *after* checking indexers because
        // the [ token can start an index signature or a computed property name
        if (scanner.tokenIsIdentifierOrKeyword(token()) || token() == SyntaxKind::StringLiteral || token() == SyntaxKind::NumericLiteral ||
            token() == SyntaxKind::AsteriskToken || token() == SyntaxKind::OpenBracketToken)
        {
            auto isAmbient = some<ModifiersArray>(modifiers, std::bind(&Parser::isDeclareModifier, this, std::placeholders::_1));
            if (isAmbient)
            {
                for (auto m : modifiers)
                {
                    (m.asMutable<Node>())->flags |= NodeFlags::Ambient;
                }
                return doInsideOfContext<Node>(NodeFlags::Ambient,
                                               [&]() { return parsePropertyOrMethodDeclaration(pos, hasJSDoc, decorators, modifiers); });
            }
            else
            {
                return parsePropertyOrMethodDeclaration(pos, hasJSDoc, decorators, modifiers);
            }
        }

        if (!!decorators || !!modifiers)
        {
            // treat this.as<a>() property declaration with a missing name.
            auto name = createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ true,
                                                      data::DiagnosticMessage(Diagnostics::Declaration_expected));
            return parsePropertyDeclaration(pos, hasJSDoc, decorators, modifiers, name, /*questionToken*/ undefined);
        }

        // 'isClassMemberStart' should have hinted not to attempt parsing.
        return Debug::fail<ClassElement>(S("Should not have attempted to parse class member declaration."));
    }

    auto parseClassExpression() -> ClassExpression
    {
        return parseClassDeclarationOrExpression(getNodePos(), hasPrecedingJSDocComment(), /*decorators*/ undefined,
                                                 /*modifiers*/ undefined, SyntaxKind::ClassExpression);
    }

    auto parseClassDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> ClassDeclaration
    {
        return parseClassDeclarationOrExpression(pos, hasJSDoc, decorators, modifiers, SyntaxKind::ClassDeclaration);
    }

    auto parseClassDeclarationOrExpression(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers,
                                           SyntaxKind kind) -> ClassLikeDeclaration
    {
        auto savedAwaitContext = inAwaitContext();
        parseExpected(SyntaxKind::ClassKeyword);
        // We don't parse the name here in await context, instead we will report a grammar error in the checker.
        auto name = parseNameOfClassDeclarationOrExpression();
        auto typeParameters = parseTypeParameters();
        if (some(modifiers, isExportModifier))
            setAwaitContext(/*value*/ true);
        auto heritageClauses = parseHeritageClauses();

        NodeArray<ClassElement> members;
        if (parseExpected(SyntaxKind::OpenBraceToken))
        {
            // ClassTail[Yield,Await] : (Modified) See 14.5
            //      ClassHeritage[?Yield,?Await]opt { ClassBody[?Yield,?Await]opt }
            members = parseClassMembers();
            parseExpected(SyntaxKind::CloseBraceToken);
        }
        else
        {
            members = createMissingList<ClassElement>();
        }
        setAwaitContext(savedAwaitContext);
        auto node = kind == SyntaxKind::ClassDeclaration
                        ? factory.createClassDeclaration(decorators, modifiers, name, typeParameters, heritageClauses, members)
                              .as<ClassLikeDeclaration>()
                        : factory.createClassExpression(decorators, modifiers, name, typeParameters, heritageClauses, members)
                              .as<ClassLikeDeclaration>();
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseNameOfClassDeclarationOrExpression() -> Identifier
    {
        // implements is a future reserved word so
        // 'class implements' might mean either
        // - class expression with omitted name, 'implements' starts heritage clause
        // - class with name 'implements'
        // 'isImplementsClause' helps to disambiguate between these two cases
        return isBindingIdentifier() && !isImplementsClause() ? createIdentifier(isBindingIdentifier()) : undefined;
    }

    auto isImplementsClause() -> boolean
    {
        return token() == SyntaxKind::ImplementsKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsIdentifierOrKeyword, this));
    }

    auto parseHeritageClauses() -> NodeArray<HeritageClause>
    {
        // ClassTail[Yield,Await] : (Modified) See 14.5
        //      ClassHeritage[?Yield,?Await]opt { ClassBody[?Yield,?Await]opt }

        if (isHeritageClause())
        {
            return parseList<HeritageClause>(ParsingContext::HeritageClauses, std::bind(&Parser::parseHeritageClause, this));
        }

        return undefined;
    }

    auto parseHeritageClause() -> HeritageClause
    {
        auto pos = getNodePos();
        auto tok = token();
        Debug::_assert(tok == SyntaxKind::ExtendsKeyword || tok == SyntaxKind::ImplementsKeyword); // isListElement() should ensure this.
        nextToken();
        auto types = parseDelimitedList<ExpressionWithTypeArguments>(ParsingContext::HeritageClauseElement,
                                                                     std::bind(&Parser::parseExpressionWithTypeArguments, this));
        return finishNode(factory.createHeritageClause(tok, types), pos);
    }

    auto parseExpressionWithTypeArguments() -> ExpressionWithTypeArguments
    {
        auto pos = getNodePos();
        auto expression = parseLeftHandSideExpressionOrHigher();
        auto typeArguments = tryParseTypeArguments();
        return finishNode(factory.createExpressionWithTypeArguments(expression, typeArguments), pos);
    }

    auto tryParseTypeArguments() -> NodeArray<TypeNode>
    {
        return token() == SyntaxKind::LessThanToken
                   ? parseBracketedList<TypeNode>(ParsingContext::TypeArguments, std::bind(&Parser::parseType, this),
                                                  SyntaxKind::LessThanToken, SyntaxKind::GreaterThanToken)
                   : undefined;
    }

    auto isHeritageClause() -> boolean
    {
        return token() == SyntaxKind::ExtendsKeyword || token() == SyntaxKind::ImplementsKeyword;
    }

    auto parseClassMembers() -> NodeArray<ClassElement>
    {
        return parseList<ClassElement>(ParsingContext::ClassMembers, std::bind(&Parser::parseClassElement, this));
    }

    auto parseInterfaceDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> InterfaceDeclaration
    {
        parseExpected(SyntaxKind::InterfaceKeyword);
        auto name = parseIdentifier();
        auto typeParameters = parseTypeParameters();
        auto heritageClauses = parseHeritageClauses();
        auto members = parseObjectTypeMembers();
        auto node = factory.createInterfaceDeclaration(decorators, modifiers, name, typeParameters, heritageClauses, members);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseTypeAliasDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> TypeAliasDeclaration
    {
        parseExpected(SyntaxKind::TypeKeyword);
        auto name = parseIdentifier();
        auto typeParameters = parseTypeParameters();
        parseExpected(SyntaxKind::EqualsToken);
        auto type =
            (token() == SyntaxKind::IntrinsicKeyword ? tryParse<TypeNode>(std::bind(&Parser::parseKeywordAndNoDot, this)) : undefined) ||
            [&]() { return parseType(); };
        parseSemicolon();
        auto node = factory.createTypeAliasDeclaration(decorators, modifiers, name, typeParameters, type);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    // In an ambient declaration, the grammar only allows integer literals.as<initializers>().
    // In a non-ambient declaration, the grammar allows uninitialized members only in a
    // ConstantEnumMemberSection, which starts at the beginning of an enum declaration
    // or any time an integer literal initializer is encountered.
    auto parseEnumMember() -> EnumMember
    {
        auto pos = getNodePos();
        auto hasJSDoc = hasPrecedingJSDocComment();
        auto name = parsePropertyName();
        auto initializer = allowInAnd<Expression>(std::bind(&Parser::parseInitializer, this));
        return withJSDoc(finishNode(factory.createEnumMember(name, initializer), pos), hasJSDoc);
    }

    auto parseEnumDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> EnumDeclaration
    {
        parseExpected(SyntaxKind::EnumKeyword);
        auto name = parseIdentifier();
        NodeArray<EnumMember> members;
        if (parseExpected(SyntaxKind::OpenBraceToken))
        {
            members = doOutsideOfYieldAndAwaitContext<NodeArray<EnumMember>>(
                [&]() { return parseDelimitedList<EnumMember>(ParsingContext::EnumMembers, std::bind(&Parser::parseEnumMember, this)); });
            parseExpected(SyntaxKind::CloseBraceToken);
        }
        else
        {
            members = createMissingList<EnumMember>();
        }
        auto node = factory.createEnumDeclaration(decorators, modifiers, name, members);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseModuleBlock() -> ModuleBlock
    {
        auto pos = getNodePos();
        NodeArray<Statement> statements;
        if (parseExpected(SyntaxKind::OpenBraceToken))
        {
            statements = parseList<Statement>(ParsingContext::BlockStatements, std::bind(&Parser::parseStatement, this));
            parseExpected(SyntaxKind::CloseBraceToken);
        }
        else
        {
            statements = createMissingList<Statement>();
        }
        return finishNode(factory.createModuleBlock(statements), pos);
    }

    auto parseModuleOrNamespaceDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers,
                                           NodeFlags flags) -> ModuleDeclaration
    {
        // If we are parsing a dotted namespace name, we want to
        // propagate the 'Namespace' flag across the names if set.
        auto namespaceFlag = flags & NodeFlags::Namespace;
        auto name = parseIdentifier();
        auto body = parseOptional(SyntaxKind::DotToken)
                        ? parseModuleOrNamespaceDeclaration(getNodePos(), /*hasJSDoc*/ false, /*decorators*/ undefined,
                                                            /*modifiers*/ undefined, NodeFlags::NestedNamespace | namespaceFlag)
                              .as<ModuleBody>()
                        : parseModuleBlock().as<ModuleBody>();
        auto node = factory.createModuleDeclaration(decorators, modifiers, name, body, flags);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseAmbientExternalModuleDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> ModuleDeclaration
    {
        auto flags = NodeFlags::None;
        LiteralLikeNode name;
        if (token() == SyntaxKind::GlobalKeyword)
        {
            // parse 'global'.as<name>() of global scope augmentation
            name = parseIdentifier();
            flags |= NodeFlags::GlobalAugmentation;
        }
        else
        {
            name = parseLiteralNode().as<StringLiteral>();
            name->text = internIdentifier(name.as<LiteralLikeNode>()->text);
        }
        ModuleBlock body;
        if (token() == SyntaxKind::OpenBraceToken)
        {
            body = parseModuleBlock();
        }
        else
        {
            parseSemicolon();
        }
        auto node = factory.createModuleDeclaration(decorators, modifiers, name, body, flags);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseModuleDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> ModuleDeclaration
    {
        NodeFlags flags = NodeFlags::None;
        if (token() == SyntaxKind::GlobalKeyword)
        {
            // global augmentation
            return parseAmbientExternalModuleDeclaration(pos, hasJSDoc, decorators, modifiers);
        }
        else if (parseOptional(SyntaxKind::NamespaceKeyword))
        {
            flags |= NodeFlags::Namespace;
        }
        else
        {
            parseExpected(SyntaxKind::ModuleKeyword);
            if (token() == SyntaxKind::StringLiteral)
            {
                return parseAmbientExternalModuleDeclaration(pos, hasJSDoc, decorators, modifiers);
            }
        }
        return parseModuleOrNamespaceDeclaration(pos, hasJSDoc, decorators, modifiers, flags);
    }

    auto isExternalModuleReference() -> boolean
    {
        return token() == SyntaxKind::RequireKeyword && lookAhead<boolean>(std::bind(&Parser::nextTokenIsOpenParen, this));
    }

    auto nextTokenIsOpenParen() -> boolean
    {
        return nextToken() == SyntaxKind::OpenParenToken;
    }

    auto nextTokenIsSlash() -> boolean
    {
        return nextToken() == SyntaxKind::SlashToken;
    }

    auto parseNamespaceExportDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> NamespaceExportDeclaration
    {
        parseExpected(SyntaxKind::AsKeyword);
        parseExpected(SyntaxKind::NamespaceKeyword);
        auto name = parseIdentifier();
        parseSemicolon();
        auto node = factory.createNamespaceExportDeclaration(name);
        // NamespaceExportDeclaration nodes cannot have decorators or modifiers, so we attach them here so we can report them in the grammar
        // checker
        node->decorators = decorators;
        copy(node->modifiers, modifiers);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseImportDeclarationOrImportEqualsDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators,
                                                         NodeArray<Modifier> modifiers) -> Node
    {
        parseExpected(SyntaxKind::ImportKeyword);

        auto afterImportPos = scanner.getStartPos();

        // We don't parse the identifier here in await context, instead we will report a grammar error in the checker.
        Identifier identifier;
        if (isIdentifier())
        {
            identifier = parseIdentifier();
        }

        auto isTypeOnly = false;
        if (token() != SyntaxKind::FromKeyword && (!!identifier && identifier->escapedText == S("type")) &&
            (isIdentifier() || tokenAfterImportDefinitelyProducesImportDeclaration()))
        {
            isTypeOnly = true;
            identifier = isIdentifier() ? parseIdentifier() : undefined;
        }

        if (!!identifier && !tokenAfterImportedIdentifierDefinitelyProducesImportDeclaration())
        {
            return parseImportEqualsDeclaration(pos, hasJSDoc, decorators, modifiers, identifier, isTypeOnly);
        }

        // ImportDeclaration:
        //  import ImportClause from ModuleSpecifier ;
        //  import ModuleSpecifier;
        ImportClause importClause;
        if (!!identifier ||                         // import id
            token() == SyntaxKind::AsteriskToken || // import *
            token() == SyntaxKind::OpenBraceToken   // import {
        )
        {
            importClause = parseImportClause(identifier, afterImportPos, isTypeOnly);
            parseExpected(SyntaxKind::FromKeyword);
        }

        auto moduleSpecifier = parseModuleSpecifier();
        parseSemicolon();
        auto node = factory.createImportDeclaration(decorators, modifiers, importClause, moduleSpecifier);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto tokenAfterImportDefinitelyProducesImportDeclaration() -> boolean
    {
        return token() == SyntaxKind::AsteriskToken || token() == SyntaxKind::OpenBraceToken;
    }

    auto tokenAfterImportedIdentifierDefinitelyProducesImportDeclaration() -> boolean
    {
        // In `import id ___`, the current token decides whether to produce
        // an ImportDeclaration or ImportEqualsDeclaration.
        return token() == SyntaxKind::CommaToken || token() == SyntaxKind::FromKeyword;
    }

    auto parseImportEqualsDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers,
                                      Identifier identifier, boolean isTypeOnly) -> ImportEqualsDeclaration
    {
        parseExpected(SyntaxKind::EqualsToken);
        auto moduleReference = parseModuleReference();
        parseSemicolon();
        auto node = factory.createImportEqualsDeclaration(decorators, modifiers, isTypeOnly, identifier, moduleReference);
        auto finished = withJSDoc(finishNode(node, pos), hasJSDoc);
        return finished;
    }

    auto parseImportClause(Identifier identifier, number pos, boolean isTypeOnly) -> ImportClause
    {
        // ImportClause:
        //  ImportedDefaultBinding
        //  NameSpaceImport
        //  NamedImports
        //  ImportedDefaultBinding, NameSpaceImport
        //  ImportedDefaultBinding, NamedImports

        // If there was no default import or if there is comma token after default import
        // parse namespace or named imports
        Node namedBindings;
        if (!identifier || parseOptional(SyntaxKind::CommaToken))
        {
            namedBindings = token() == SyntaxKind::AsteriskToken ? parseNamespaceImport().as<Node>()
                                                                 : parseNamedImportsOrExports(SyntaxKind::NamedImports).as<Node>();
        }

        return finishNode(factory.createImportClause(isTypeOnly, identifier, namedBindings), pos);
    }

    auto parseModuleReference() -> Node
    {
        return isExternalModuleReference() ? parseExternalModuleReference().as<Node>()
                                           : parseEntityName(/*allowReservedWords*/ false).as<Node>();
    }

    auto parseExternalModuleReference() -> ExternalModuleReference
    {
        auto pos = getNodePos();
        parseExpected(SyntaxKind::RequireKeyword);
        parseExpected(SyntaxKind::OpenParenToken);
        auto expression = parseModuleSpecifier();
        parseExpected(SyntaxKind::CloseParenToken);
        return finishNode(factory.createExternalModuleReference(expression), pos);
    }

    auto parseModuleSpecifier() -> Expression
    {
        if (token() == SyntaxKind::StringLiteral)
        {
            auto result = parseLiteralNode();
            result->text = internIdentifier(result->text);
            return result;
        }
        else
        {
            // We allow arbitrary expressions here, even though the grammar only allows string
            // literals.  We check to ensure that it is only a string literal later in the grammar
            // check pass.
            return parseExpression();
        }
    }

    auto parseNamespaceImport() -> NamespaceImport
    {
        // NameSpaceImport:
        //  *.as<ImportedBinding>()
        auto pos = getNodePos();
        parseExpected(SyntaxKind::AsteriskToken);
        parseExpected(SyntaxKind::AsKeyword);
        auto name = parseIdentifier();
        return finishNode(factory.createNamespaceImport(name), pos);
    }

    auto parseNamedImportsOrExports(SyntaxKind kind) -> NamedImportsOrExports
    {
        auto pos = getNodePos();

        // NamedImports:
        //  { }
        //  { ImportsList }
        //  { ImportsList, }

        // ImportsList:
        //  ImportSpecifier
        //  ImportsList, ImportSpecifier
        auto node =
            kind == SyntaxKind::NamedImports
                ? factory
                      .createNamedImports(parseBracketedList<ImportSpecifier>(ParsingContext::ImportOrExportSpecifiers,
                                                                              std::bind(&Parser::parseImportSpecifier, this),
                                                                              SyntaxKind::OpenBraceToken, SyntaxKind::CloseBraceToken))
                      .as<NamedImportsOrExports>()
                : factory
                      .createNamedExports(parseBracketedList<ExportSpecifier>(ParsingContext::ImportOrExportSpecifiers,
                                                                              std::bind(&Parser::parseExportSpecifier, this),
                                                                              SyntaxKind::OpenBraceToken, SyntaxKind::CloseBraceToken))
                      .as<NamedImportsOrExports>();
        return finishNode(node, pos);
    }

    auto parseExportSpecifier() -> ImportOrExportSpecifier
    {
        return parseImportOrExportSpecifier(SyntaxKind::ExportSpecifier).as<ExportSpecifier>();
    }

    auto parseImportSpecifier() -> ImportOrExportSpecifier
    {
        return parseImportOrExportSpecifier(SyntaxKind::ImportSpecifier).as<ImportSpecifier>();
    }

    auto parseImportOrExportSpecifier(SyntaxKind kind) -> ImportOrExportSpecifier
    {
        auto pos = getNodePos();
        // ImportSpecifier:
        //   BindingIdentifier
        //   IdentifierName.as<BindingIdentifier>()
        // ExportSpecifier:
        //   IdentifierName
        //   IdentifierName.as<IdentifierName>()
        auto checkIdentifierIsKeyword = isKeyword(token()) && !isIdentifier();
        auto checkIdentifierStart = scanner.getTokenPos();
        auto checkIdentifierEnd = scanner.getTextPos();
        auto identifierName = parseIdentifierName();
        Identifier propertyName;
        Identifier name;
        if (token() == SyntaxKind::AsKeyword)
        {
            propertyName = identifierName;
            parseExpected(SyntaxKind::AsKeyword);
            checkIdentifierIsKeyword = isKeyword(token()) && !isIdentifier();
            checkIdentifierStart = scanner.getTokenPos();
            checkIdentifierEnd = scanner.getTextPos();
            name = parseIdentifierName();
        }
        else
        {
            name = identifierName;
        }
        if (kind == SyntaxKind::ImportSpecifier && checkIdentifierIsKeyword)
        {
            parseErrorAt(checkIdentifierStart, checkIdentifierEnd, data::DiagnosticMessage(Diagnostics::Identifier_expected));
        }
        auto node = kind == SyntaxKind::ImportSpecifier ? factory.createImportSpecifier(propertyName, name).as<ImportOrExportSpecifier>()
                                                        : factory.createExportSpecifier(propertyName, name).as<ImportOrExportSpecifier>();
        return finishNode(node, pos);
    }

    auto parseNamespaceExport(number pos) -> NamespaceExport
    {
        return finishNode(factory.createNamespaceExport(parseIdentifierName()), pos);
    }

    auto parseExportDeclaration(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> ExportDeclaration
    {
        auto savedAwaitContext = inAwaitContext();
        setAwaitContext(/*value*/ true);
        NamedExportBindings exportClause;
        Expression moduleSpecifier;
        auto isTypeOnly = parseOptional(SyntaxKind::TypeKeyword);
        auto namespaceExportPos = getNodePos();
        if (parseOptional(SyntaxKind::AsteriskToken))
        {
            if (parseOptional(SyntaxKind::AsKeyword))
            {
                exportClause = parseNamespaceExport(namespaceExportPos);
            }
            parseExpected(SyntaxKind::FromKeyword);
            moduleSpecifier = parseModuleSpecifier();
        }
        else
        {
            exportClause = parseNamedImportsOrExports(SyntaxKind::NamedExports);
            // It is not uncommon to accidentally omit the 'from' keyword. Additionally, in editing scenarios,
            // the 'from' keyword can be parsed.as<a>() named when the clause is unterminated (i.e. `{ from "moduleName";`)
            // If we don't have a 'from' keyword, see if we have a string literal such that ASI won't take effect.
            if (token() == SyntaxKind::FromKeyword || (token() == SyntaxKind::StringLiteral && !scanner.hasPrecedingLineBreak()))
            {
                parseExpected(SyntaxKind::FromKeyword);
                moduleSpecifier = parseModuleSpecifier();
            }
        }
        parseSemicolon();
        setAwaitContext(savedAwaitContext);
        auto node = factory.createExportDeclaration(decorators, modifiers, isTypeOnly, exportClause, moduleSpecifier);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto parseExportAssignment(number pos, boolean hasJSDoc, NodeArray<Decorator> decorators, NodeArray<Modifier> modifiers)
        -> ExportAssignment
    {
        auto savedAwaitContext = inAwaitContext();
        setAwaitContext(/*value*/ true);
        auto isExportEquals = false;
        if (parseOptional(SyntaxKind::EqualsToken))
        {
            isExportEquals = true;
        }
        else
        {
            parseExpected(SyntaxKind::DefaultKeyword);
        }
        auto expression = parseAssignmentExpressionOrHigher();
        parseSemicolon();
        setAwaitContext(savedAwaitContext);
        auto node = factory.createExportAssignment(decorators, modifiers, isExportEquals, expression);
        return withJSDoc(finishNode(node, pos), hasJSDoc);
    }

    auto setExternalModuleIndicator(SourceFile sourceFile) -> void
    {
        // Try to use the first top-level import/when available, then
        // fall back to looking for an 'import.meta' somewhere in the tree if necessary.
        sourceFile->externalModuleIndicator =
            forEach<decltype(sourceFile->statements), Node>(
                sourceFile->statements, (FuncT<Node>)std::bind(&Parser::isAnExternalModuleIndicatorNode, this, std::placeholders::_1)) ||
            [&]() { return getImportMetaIfNecessary(sourceFile); };
    }

    auto isAnExternalModuleIndicatorNode(Node node) -> Node
    {
        return hasModifierOfKind(node, SyntaxKind::ExportKeyword) ||
                       isImportEqualsDeclaration(node) &&
                           ts::isExternalModuleReference(node.as<ImportEqualsDeclaration>()->moduleReference) ||
                       isImportDeclaration(node) || isExportAssignment(node) || isExportDeclaration(node)
                   ? node
                   : undefined;
    }

    auto getImportMetaIfNecessary(SourceFile sourceFile) -> Node
    {
        return !!(sourceFile->flags & NodeFlags::PossiblyContainsImportMeta) ? walkTreeForExternalModuleIndicators(sourceFile) : undefined;
    }

    auto walkTreeForExternalModuleIndicators(Node node) -> Node
    {
        return isImportMeta(node)
                   ? node
                   : forEachChild<Node, Node>(node, std::bind(&Parser::walkTreeForExternalModuleIndicators, this, std::placeholders::_1));
    }

    /** Do not use hasModifier inside the parser; it relies on parent pointers. Use this instead. */
    auto hasModifierOfKind(Node node, SyntaxKind kind) -> boolean
    {
        return some(node->modifiers, [=](auto m) { return m == kind; });
    }

    auto isImportMeta(Node node) -> boolean
    {
        return isMetaProperty(node) && node.as<MetaProperty>()->keywordToken == SyntaxKind::ImportKeyword &&
               node.as<MetaProperty>()->name->escapedText == S("meta");
    }

    // [[[ namespace JSDocParser ]]]

    auto parseJSDocTypeExpressionForTests(string content, number start, number length) -> NodeWithDiagnostics
    {
        initializeState(S("file.js"), content, ScriptTarget::Latest, /*_syntaxCursor:*/ undefined, ScriptKind::JS);
        scanner.setText(content, start, length);
        currentToken = scanner.scan();
        auto jsDocTypeExpression = parseJSDocTypeExpression();

        auto sourceFile = createSourceFile(S("file.js"), ScriptTarget::Latest, ScriptKind::JS, /*isDeclarationFile*/ false,
                                           NodeArray<Statement>(), factory.createToken(SyntaxKind::EndOfFileToken), NodeFlags::None);
        auto diagnostics = attachFileToDiagnostics(parseDiagnostics, sourceFile);
        if (!!jsDocDiagnostics)
        {
            copy(sourceFile->jsDocDiagnostics, attachFileToDiagnostics(jsDocDiagnostics, sourceFile));
        }

        clearState();

        if (!!jsDocTypeExpression)
        {
            NodeWithDiagnostics nodeWithDiagnostics;
            nodeWithDiagnostics->node = jsDocTypeExpression;
            copy(nodeWithDiagnostics->diagnostics, diagnostics);
        }

        return undefined;
    }

    // Parses out a JSDoc type expression.
    auto parseJSDocTypeExpression(boolean mayOmitBraces = false) -> JSDocTypeExpression
    {
        auto pos = getNodePos();
        auto hasBrace = mayOmitBraces ? parseOptional(SyntaxKind::OpenBraceToken) : parseExpected(SyntaxKind::OpenBraceToken);
        auto type = doInsideOfContext<TypeNode>(NodeFlags::JSDoc, std::bind(&Parser::parseJSDocType, this));
        if (!mayOmitBraces || hasBrace)
        {
            parseExpectedJSDoc(SyntaxKind::CloseBraceToken);
        }

        auto result = factory.createJSDocTypeExpression(type);
        fixupParentReferences(result);
        return finishNode(result, pos);
    }

    auto parseJSDocNameReference() -> JSDocNameReference
    {
        auto pos = getNodePos();
        auto hasBrace = parseOptional(SyntaxKind::OpenBraceToken);
        auto entityName = parseEntityName(/* allowReservedWords*/ false);
        if (hasBrace)
        {
            parseExpectedJSDoc(SyntaxKind::CloseBraceToken);
        }

        auto result = factory.createJSDocNameReference(entityName);
        fixupParentReferences(result);
        return finishNode(result, pos);
    }

    auto parseIsolatedJSDocComment(string content, number start, number length) -> NodeWithDiagnostics
    {
        initializeState(string(), content, ScriptTarget::Latest, /*_syntaxCursor:*/ undefined, ScriptKind::JS);
        auto jsDoc = doInsideOfContext<JSDoc>(NodeFlags::JSDoc, [&]() { return parseJSDocCommentWorker(start, length); });

        auto sourceFile = SourceFile();
        sourceFile->text = content;
        sourceFile->languageVariant = LanguageVariant::Standard;
        auto diagnostics = attachFileToDiagnostics(parseDiagnostics, sourceFile);
        clearState();

        if (!!jsDoc)
        {
            auto nodeWithDiagnostics = NodeWithDiagnostics();
            nodeWithDiagnostics->node = jsDoc;
            copy(nodeWithDiagnostics->diagnostics, diagnostics);
        }

        return undefined;
    }

    auto parseJSDocComment(Node parent, number start, number length) -> JSDoc
    {
        auto saveToken = currentToken;
        // TODO: does it make any sense
        // auto saveParseDiagnosticsLength = parseDiagnostics.size();
        auto saveParseErrorBeforeNextFinishedNode = parseErrorBeforeNextFinishedNode;

        auto comment = doInsideOfContext<JSDoc>(NodeFlags::JSDoc, [&]() { return parseJSDocCommentWorker(start, length); });
        setParent(comment, parent);

        if (!!(contextFlags & NodeFlags::JavaScriptFile))
        {
            if (!jsDocDiagnostics.empty())
            {
                jsDocDiagnostics.clear();
            }
            copy(jsDocDiagnostics, parseDiagnostics);
        }
        currentToken = saveToken;
        // TODO: does it make any sense
        // parseDiagnostics->length = saveParseDiagnosticsLength;
        parseErrorBeforeNextFinishedNode = saveParseErrorBeforeNextFinishedNode;
        return comment;
    }

    auto parseJSDocCommentWorker(number start = 0, number length = -1) -> JSDoc
    {
        // TODO: finish it
        // ParseJSDocCommentClass p(scanner, this, sourceText);
        // return p.parseJSDocCommentWorker(start, length);
        return JSDoc();
    } // end of parseJSDocCommentWorker

  public:
    auto createSourceFile(string fileName, string sourceText, ScriptTarget languageVersion, boolean setParentNodes = false,
                          ScriptKind scriptKind = ScriptKind::Unknown) -> SourceFile
    {
        SourceFile result;
        if (languageVersion == ScriptTarget::JSON)
        {
            result = parseSourceFile(fileName, sourceText, languageVersion, undefined /*syntaxCursor*/, setParentNodes, ScriptKind::JSON);
        }
        else
        {
            result = parseSourceFile(fileName, sourceText, languageVersion, undefined /*syntaxCursor*/, setParentNodes, scriptKind);
        }

        return result;
    }

    /* @internal */
    auto _parseIsolatedJSDocComment(string content, number start, number length) -> NodeWithDiagnostics
    {
        auto result = parseIsolatedJSDocComment(content, start, length);
        if (!!result && !!result->jsDoc)
        {
            // because the jsDocComment was parsed out of the source file, it might
            // not be covered by the fixupParentReferences.
            fixupParentReferences(result->jsDoc);
        }

        return result;
    }

    /*@internal*/
    struct Pair
    {
        string name;
        string _args;
    };

    auto processCommentPragmas(SourceFile context, string sourceText) -> void
    {
        std::vector<Pair> pragmas;

        for (auto &range : scanner.getLeadingCommentRanges(sourceText, 0))
        {
            auto comment = safe_string(sourceText).substring(range->pos, range->_end);
            extractPragmas(pragmas, range, comment);
        }

        context->pragmas.clear();
        for (auto &pragma : pragmas)
        {
            if (context->pragmas.find(pragma.name) != context->pragmas.end())
            {
                auto currentValue = context->pragmas.at(pragma.name);
                context->pragmas[pragma.name] = currentValue + S(";") + pragma._args;
                continue;
            }
            context->pragmas[pragma.name] = pragma._args;
        }
    }

    /*@internal*/
    auto processPragmasIntoFields(SourceFile context, PragmaDiagnosticReporter reportDiagnostic) -> void
    {
        context->checkJsDirective = undefined;
        context->referencedFiles.clear();
        context->typeReferenceDirectives.clear();
        context->libReferenceDirectives.clear();
        context->amdDependencies.clear();
        context->hasNoDefaultLib = false;
        for (auto &pair : context->pragmas)
        {
            auto entryOrList = pair.first;
            auto key = pair.second;

            static std::map<string, int> cases = {{S("reference"), 1},  {S("amd-dependency"), 2},  {S("amd-module"), 3},
                                                  {S("ts-nocheck"), 4}, {S("ts-check"), 5},        {S("jsx"), 6},
                                                  {S("jsxfrag"), 7},    {S("jsximportsource"), 8}, {S("jsxruntime"), 9}};

            /*JSDocTag*/ Node tag;
            auto index = cases[key];
            switch (index)
            {
            case 1: {
                auto referencedFiles = context->referencedFiles;
                auto typeReferenceDirectives = context->typeReferenceDirectives;
                auto libReferenceDirectives = context->libReferenceDirectives;
                // TODO...
                break;
            }
            case 2: {
                // TODO...
                break;
            }
            case 3: {
                // TODO...
                break;
            }
            case 4:
            case 5: {
                // TODO...
                break;
            }
            case 6:
            case 7:
            case 8:
            case 9:
                return; // Accessed directly
            default:
                Debug::fail<void>(S("Unhandled pragma kind")); // Can this be made into an assertNever in the future?
            }
        }
    }

    // std::map<string, regex> namedArgRegExCache;
    // auto getNamedArgRegEx(string name) -> regex {
    //     if (namedArgRegExCache.find(name) != namedArgRegExCache.end()) {
    //         return namedArgRegExCache.at(name);
    //     }
    //     regex result(S("(\\s${name}\\s*=\\s*)('|\")(.+?)\\2"), std::regex_constants::extended|std::regex_constants::icase);
    //     namedArgRegExCache[name] = result;
    //     return result;
    // }

    // regex tripleSlashXMLCommentStartRegEx = regex(S("^\\/\\/\\/\\s*<(\\S+)\\s.*?\\/"),
    // std::regex_constants::extended|std::regex_constants::icase); regex singleLinePragmaRegEx =
    // regex(S("^\\/\\/\\/?\\s*@(\\S+)\\s*(.*)\\s*$"), std::regex_constants::extended|std::regex_constants::icase); regex
    // multiLinePragmaRegEx = regex(S("\\s*@(\\S+)\\s*(.*)\\s*$"), std::regex_constants::extended|std::regex_constants::icase); // Defined
    // inline since it uses the "g" flag, which keeps a persistent index (for iterating)
    auto extractPragmas(std::vector<Pair> pragmas, CommentRange range, string text) -> void
    {
        // auto tripleSlash = range->kind == SyntaxKind::SingleLineCommentTrivia && regex_exec(text, tripleSlashXMLCommentStartRegEx);
        // if (tripleSlash) {
        //}

        // TODO: complete later
    }

    // auto addPragmaForMatch(std::vector<PragmaPseudoMapEntry> pragmas, CommentRange range, PragmaKindFlags kind, RegExpExecArray match) {
    //     if (!match) return;
    //     auto name = match[1].toLowerCase().as<keyof>() PragmaPseudoMap; // Technically unsafe cast, but we do it so they below check to
    //     make it safe typechecks auto pragma = commentPragmas[name].as<PragmaDefinition>(); if (!pragma || !(pragma->kind! & kind)) {
    //         return;
    //     }
    //     auto args = match[2]; // Split on spaces and match up positionally with definition
    //     auto argument = getNamedPragmaArguments(pragma, args);
    //     if (argument == "fail") return; // Missing required argument, fail to parse it
    //     pragmas.push({ name, args: { argument arguments, range } }.as<PragmaPseudoMapEntry>());
    //     return;
    // }

    // auto getNamedPragmaArguments(PragmaDefinition pragma, string text) -> std::map<string, string> {
    //     if (!text) return {};
    //     if (!pragma.args) return {};
    //     auto args = text.split(regex(S("\\s+")));
    //     auto argMap: {[string index]: string} = {};
    //     for (auto i = 0; i < pragma.args.size(); i++) {
    //         auto argument = pragma.args[i];
    //         if (!args[i] && !argument.optional) {
    //             return "fail";
    //         }
    //         if (argument.captureSpan) {
    //             return Debug::fail("Capture spans not yet implemented for non-xml pragmas");
    //         }
    //         argMap[argument.name] = args[i];
    //     }
    //     return argMap;
    // }

    /** @internal */
    auto tagNamesAreEquivalent(JsxTagNameExpression lhs, JsxTagNameExpression rhs) -> boolean
    {
        if (lhs != rhs)
        {
            return false;
        }

        if (lhs == SyntaxKind::Identifier)
        {
            return lhs.as<Identifier>()->escapedText == rhs.as<Identifier>()->escapedText;
        }

        if (lhs == SyntaxKind::ThisKeyword)
        {
            return true;
        }

        // If we are at this statement then we must have PropertyAccessExpression and because tag name in Jsx element can only
        // take forms of JsxTagNameExpression which includes an identifier, "this" expression, or another propertyAccessExpression
        // it is safe to case the expression property.as<such>(). See parseJsxElementName for how we parse tag name in Jsx element

        auto lhsName = ts::isIdentifier(lhs.as<PropertyAccessExpression>()->name)
                           ? lhs.as<PropertyAccessExpression>()->name.as<Identifier>()->escapedText
                       : isPrivateIdentifier(lhs.as<PropertyAccessExpression>()->name)
                           ? lhs.as<PropertyAccessExpression>()->name.as<PrivateIdentifier>()->escapedText
                           : string();
        auto rhsName = ts::isIdentifier(rhs.as<PropertyAccessExpression>()->name)
                           ? rhs.as<PropertyAccessExpression>()->name.as<Identifier>()->escapedText
                       : isPrivateIdentifier(rhs.as<PropertyAccessExpression>()->name)
                           ? rhs.as<PropertyAccessExpression>()->name.as<PrivateIdentifier>()->escapedText
                           : string();
        return lhsName == rhsName && tagNamesAreEquivalent(lhs.as<PropertyAccessExpression>()->expression.as<JsxTagNameExpression>(),
                                                           rhs.as<PropertyAccessExpression>()->expression.as<JsxTagNameExpression>());
    }

}; // End of Scanner

// // Produces a new SourceFile for the 'newText' provided. The 'textChangeRange' parameter
// // indicates what changed between the 'text' that this SourceFile has and the 'newText'.
// // The SourceFile will be created with the compiler attempting to reuse.as<many>() nodes from
// // this file.as<possible>().
// //
// // this Note auto mutates nodes from this SourceFile. That means any existing nodes
// // from this SourceFile that are being held onto may change.as<a>() result (including
// // becoming detached from any SourceFile).  It is recommended that this SourceFile not
// // be used once 'update' is called on it.
// auto updateSourceFile(SourceFile sourceFile, string newText, TextChangeRange textChangeRange, boolean aggressiveChecks = false) ->
// SourceFile {
//     auto newSourceFile = IncrementalParser::updateSourceFile(sourceFile, newText, textChangeRange, aggressiveChecks);
//     // Because new source file node is created, it may not have the flag PossiblyContainDynamicImport. This is the case if there is no
//     new edit to add dynamic import.
//     // We will manually port the flag to the new source file.
//     newSourceFile->flags |= (sourceFile->flags & NodeFlags::PermanentlySetIncrementalFlags);
//     return newSourceFile;
// }
} // namespace Impl

// See also `isExternalOrCommonJsModule` in utilities.ts
auto isExternalModule(SourceFile file) -> boolean
{
    return !!file->externalModuleIndicator;
}

Parser::Parser()
{
    impl = new ts::Impl::Parser();
}

auto Parser::parseSourceFile(string sourceText, ScriptTarget languageVersion) -> SourceFile
{
    return impl->parseSourceFile(string(), sourceText, languageVersion, IncrementalParser::SyntaxCursor());
}

auto Parser::parseSourceFile(string fileName, string sourceText, ScriptTarget languageVersion) -> SourceFile
{
    return impl->parseSourceFile(fileName, sourceText, languageVersion, IncrementalParser::SyntaxCursor());
}

auto Parser::parseSourceFile(string fileName, string sourceText, ScriptTarget languageVersion, IncrementalParser::SyntaxCursor syntaxCursor,
                             boolean setParentNodes, ScriptKind scriptKind) -> SourceFile
{
    return impl->parseSourceFile(fileName, sourceText, languageVersion, syntaxCursor, setParentNodes, scriptKind);
}

auto Parser::tokenToText(SyntaxKind kind) -> string
{
    return impl->scanner.tokenToString(kind);
}

auto Parser::syntaxKindString(SyntaxKind kind) -> string
{
    return impl->scanner.syntaxKindString(kind);
}

auto Parser::getLineAndCharacterOfPosition(SourceFileLike sourceFile, number position) -> LineAndCharacter
{
    return impl->scanner.getLineAndCharacterOfPosition(sourceFile, position);
}

Parser::~Parser()
{
    delete impl;
}

// TODO: temporary solution
namespace IncrementalParser
{
auto createSyntaxCursor(SourceFile sourceFile) -> SyntaxCursor
{
    return SyntaxCursor();
}
} // namespace IncrementalParser
} // namespace ts
