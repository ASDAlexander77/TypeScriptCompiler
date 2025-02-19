#include "parser_jdoc.h"
#include "scanner.h"
#include "parser.h"

struct ParseJSDocCommentClass
{
    string content;
    number start;
    number end;

    NodeArray<JSDocTag> tags;
    number tagsPos;
    number tagsEnd;
    std::vector<string> comments;

    Scanner &scanner;
    Parser *parser;

    ParseJSDocCommentClass(Scanner &scanner, Parser* parser, string sourceText) : content(sourceText), scanner(scanner)
    {
    }

    /*@internal*/
    auto isJSDocLikeText(safe_string text, number start)
    {
        return text[start + 1] == CharacterCodes::asterisk &&
                text[start + 2] == CharacterCodes::asterisk &&
                text[start + 3] != CharacterCodes::slash;
    }

    auto parseJSDocCommentWorker(number start = 0, number length = -1) -> JSDoc {
        end = length == -1 ? content.size() : start + length;
        length = end - start;

        Debug::_assert(start >= 0);
        Debug::_assert(start <= end);
        Debug::_assert(end <= content.size());

        // Check for /** (JSDoc opening part)
        if (!isJSDocLikeText(content, start)) {
            return undefined;
        }

        // + 3 for leading /**, - 5 in total for /** */
        return scanner.scanRange<JSDoc>(start + 3, length - 5, [&] () {
            // Initially we can parse out a tag.  We also have seen a starting asterisk.
            // This is so that /** * @type */ doesn't parse.
            auto state = JSDocState::SawAsterisk;
            number margin;
            // + 4 for leading '/** '
            // + 1 because the last index of \n is always one index before the first character in the line and coincidentally, if there is no \n before start, it is -1, which is also one index before the first character
            auto indent = start - (content.find_last_of(S('\n'), start) + 1) + 4;
            auto pushComment = [&](string text) {
                if (!margin) {
                    margin = indent;
                }
                comments.push_back(text);
                indent += text.size();
            };

            parser->nextTokenJSDoc();
            while (parseOptionalJsdoc(SyntaxKind::WhitespaceTrivia));
            if (parseOptionalJsdoc(SyntaxKind::NewLineTrivia)) {
                state = JSDocState::BeginningOfLine;
                indent = 0;
            }
            while (true) {
                switch (parser->token()) {
                    case SyntaxKind::AtToken:
                        if (state == JSDocState::BeginningOfLine || state == JSDocState::SawAsterisk) {
                            removeTrailingWhitespace(comments);
                            addTag(parseTag(indent));
                            // According NOTE to usejsdoc.org, a tag goes to end of line, except the last tag.
                            // Real-world comments may break this rule, so "BeginningOfLine" will not be a real line beginning
                            // for malformed examples like `/** @param {string} x @returns {number} the length */`
                            state = JSDocState::BeginningOfLine;
                            margin = -1;
                        }
                        else {
                            pushComment(scanner.getTokenText());
                        }
                        break;
                    case SyntaxKind::NewLineTrivia:
                        comments.push_back(scanner.getTokenText());
                        state = JSDocState::BeginningOfLine;
                        indent = 0;
                        break;
                    case SyntaxKind::AsteriskToken:
                        {
                            auto asterisk = scanner.getTokenText();
                            if (state == JSDocState::SawAsterisk || state == JSDocState::SavingComments) {
                                // If we've already seen an asterisk, then we can no longer parse a tag on this line
                                state = JSDocState::SavingComments;
                                pushComment(asterisk);
                            }
                            else {
                                // Ignore the first asterisk on a line
                                state = JSDocState::SawAsterisk;
                                indent += asterisk.size();
                            }
                        }
                        break;
                    case SyntaxKind::WhitespaceTrivia:
                        {
                            // only collect whitespace if we're already saving comments or have just crossed the comment indent margin
                            auto whitespace = scanner.getTokenText();
                            if (state == JSDocState::SavingComments) {
                                comments.push_back(whitespace);
                            }
                            else if (margin != -1 && indent + whitespace.size() > margin) {
                                comments.push_back(whitespace.substr(margin - indent));
                            }
                            indent += whitespace.size();
                        }
                        break;
                    case SyntaxKind::EndOfFileToken:
                        goto loop;
                    default:
                        // Anything else is doc comment text. We just save it. Because it
                        // wasn't a tag, we can no longer parse a tag on this line until we hit the next
                        // line break.
                        state = JSDocState::SavingComments;
                        pushComment(scanner.getTokenText());
                        break;
                }
                parser->nextTokenJSDoc();
            }
            loop:
            removeLeadingNewlines(comments);
            removeTrailingWhitespace(comments);
            return createJSDocComment();
        }); // end of lambda
    }

    auto removeLeadingNewlines(std::vector<string> comments) -> void {
        while (comments.size() && (comments[0] == S("\n") || comments[0] == S("\r"))) {
            comments.erase(comments.begin());
        }
    }

    auto removeTrailingWhitespace(std::vector<string> comments) -> void {
        while (comments.size() && trim(comments[comments.size() - 1]) == string()) {
            comments.erase(comments.end());
        }
    }

    auto createJSDocComment() -> JSDoc {
        auto comment = comments.size() ? join(comments) : string();
        auto tagsArray = !!tags ? tags : parser->createNodeArray(tags, tagsPos, tagsEnd);
        return parser->finishNode(parser->factory.createJSDocComment(comment, tagsArray), start, end);
    }

    auto isNextNonwhitespaceTokenEndOfFile() -> boolean {
        // We must use infinite lookahead,.as<there>() could be any number of newlines :(
        while (true) {
            parser->nextTokenJSDoc();
            if (parser->token() == SyntaxKind::EndOfFileToken) {
                return true;
            }
            if (!(parser->token() == SyntaxKind::WhitespaceTrivia || parser->token() == SyntaxKind::NewLineTrivia)) {
                return false;
            }
        }
    }

    auto skipWhitespace() -> void {
        if (parser->token() == SyntaxKind::WhitespaceTrivia || parser->token() == SyntaxKind::NewLineTrivia) {
            if (parser->lookAhead<boolean>(std::bind(&ParseJSDocCommentClass::isNextNonwhitespaceTokenEndOfFile, this))) {
                return; // Don't skip whitespace prior to EoF (or end of comment) - that shouldn't be included in any node's range
            }
        }
        while (parser->token() == SyntaxKind::WhitespaceTrivia || parser->token() == SyntaxKind::NewLineTrivia) {
            parser->nextTokenJSDoc();
        }
    }

    auto skipWhitespaceOrAsterisk() -> string {
        if (parser->token() == SyntaxKind::WhitespaceTrivia || parser->token() == SyntaxKind::NewLineTrivia) {
            if (parser->lookAhead<boolean>(std::bind(&ParseJSDocCommentClass::isNextNonwhitespaceTokenEndOfFile, this))) {
                return string(); // Don't skip whitespace prior to EoF (or end of comment) - that shouldn't be included in any node's range
            }
        }

        auto precedingLineBreak = scanner.hasPrecedingLineBreak();
        auto seenLineBreak = false;
        auto indentText = string();
        while ((precedingLineBreak && parser->token() == SyntaxKind::AsteriskToken) || parser->token() == SyntaxKind::WhitespaceTrivia || parser->token() == SyntaxKind::NewLineTrivia) {
            indentText += scanner.getTokenText();
            if (parser->token() == SyntaxKind::NewLineTrivia) {
                precedingLineBreak = true;
                seenLineBreak = true;
                indentText = string();
            }
            else if (parser->token() == SyntaxKind::AsteriskToken) {
                precedingLineBreak = false;
            }
            parser->nextTokenJSDoc();
        }
        return seenLineBreak ? indentText : string();
    }

    auto parseTag(number margin) -> Node {
        Debug::_assert(parser->token() == SyntaxKind::AtToken);
        auto start = scanner.getTokenPos();
        parser->nextTokenJSDoc();

        auto tagName = parseJSDocIdentifierName(/*message*/ undefined);
        auto indentText = skipWhitespaceOrAsterisk();

        static std::map<string, int> m = {{S("author"), 1}, {S("implements"), 2}, {S("augments"), 3}, {S("extends"), 4}, {S("class"), 5}, 
            {S("constructor"), 6}, {S("public"), 7}, {S("private"), 8}, {S("protected"), 9}, {S("readonly"), 10}, {S("deprecated"), 11}, 
            {S("this"), 12}, {S("enum"), 13}, {S("arg"), 14}, {S("argument"), 15}, {S("param"), 16}, {S("return"), 17}, {S("returns"), 18}, 
            {S("template"), 19}, {S("type"), 20}, {S("typedef"), 21}, {S("callback"), 22}, {S("see"), 23}};

        /*JSDocTag*/Node tag;
        auto index = m[tagName->escapedText];
        switch (index) {
            case 1:
                tag = parseAuthorTag(start, tagName, margin, indentText);
                break;
            case 2:
                tag = parseImplementsTag(start, tagName, margin, indentText);
                break;
            case 3:
            case 4:
                tag = parseAugmentsTag(start, tagName, margin, indentText);
                break;
            case 5:
            case 6:
                tag = parseSimpleTag(start, std::bind(&NodeFactory::createJSDocClassTag, &parser->factory, std::placeholders::_1, std::placeholders::_2), tagName, margin, indentText);
                break;
            case 7:
                tag = parseSimpleTag(start, std::bind(&NodeFactory::createJSDocPublicTag, &parser->factory, std::placeholders::_1, std::placeholders::_2), tagName, margin, indentText);
                break;
            case 8:
                tag = parseSimpleTag(start, std::bind(&NodeFactory::createJSDocPrivateTag, &parser->factory, std::placeholders::_1, std::placeholders::_2), tagName, margin, indentText);
                break;
            case 9:
                tag = parseSimpleTag(start, std::bind(&NodeFactory::createJSDocProtectedTag, &parser->factory, std::placeholders::_1, std::placeholders::_2), tagName, margin, indentText);
                break;
            case 10:
                tag = parseSimpleTag(start, std::bind(&NodeFactory::createJSDocReadonlyTag, &parser->factory, std::placeholders::_1, std::placeholders::_2), tagName, margin, indentText);
                break;
            case 11:
                parser->hasDeprecatedTag = true;
                tag = parseSimpleTag(start, std::bind(&NodeFactory::createJSDocDeprecatedTag, &parser->factory, std::placeholders::_1, std::placeholders::_2), tagName, margin, indentText);
                break;
            case 12:
                tag = parseThisTag(start, tagName, margin, indentText);
                break;
            case 13:
                tag = parseEnumTag(start, tagName, margin, indentText);
                break;
            case 14:
            case 15:
            case 16:
                return parseParameterOrPropertyTag(start, tagName, PropertyLikeParse::Parameter, margin);
            case 17:
            case 18:
                tag = parseReturnTag(start, tagName, margin, indentText);
                break;
            case 19:
                tag = parseTemplateTag(start, tagName, margin, indentText);
                break;
            case 20:
                tag = parseTypeTag(start, tagName, margin, indentText);
                break;
            case 21:
                tag = parseTypedefTag(start, tagName, margin, indentText);
                break;
            case 22:
                tag = parseCallbackTag(start, tagName, margin, indentText);
                break;
            case 23:
                tag = parseSeeTag(start, tagName, margin, indentText);
                break;
            default:
                tag = parseUnknownTag(start, tagName, margin, indentText);
                break;
        }
        return tag;
    }

    auto parseTrailingTagComments(pos_type pos, number end, number margin, string indentText) {
        // some tags, like typedef and callback, have already parsed their comments earlier
        if (!indentText.empty()) {
            margin += end - pos;
        }
        return parseTagComments(margin, indentText.substr(margin));
    }

    auto parseTagComments(number indent, string initialMargin) -> string {
        std::vector<string> comments;
        auto state = JSDocState::BeginningOfLine;
        auto previousWhitespace = true;
        number margin;
        auto pushComment = [&](string text) {
            if (!margin) {
                margin = indent;
            }
            comments.push_back(text);
            indent += text.size();
        };
        if (!initialMargin.empty()) {
            // jump straight to saving comments if there is some initial indentation
            if (initialMargin != string()) {
                pushComment(initialMargin);
            }
            state = JSDocState::SawAsterisk;
        }
        auto tok = parser->token();
        while (true) {
            switch (tok) {
                case SyntaxKind::NewLineTrivia:
                    state = JSDocState::BeginningOfLine;
                    // don't use pushComment here because we want to keep the margin unchanged
                    comments.push_back(scanner.getTokenText());
                    indent = 0;
                    break;
                case SyntaxKind::AtToken:
                    if (state == JSDocState::SavingBackticks || !previousWhitespace && state == JSDocState::SavingComments) {
                        // @ doesn't start a new tag inside ``, and inside a comment, only after whitespace
                        comments.push_back(scanner.getTokenText());
                        break;
                    }
                    scanner.setTextPos(scanner.getTextPos() - 1);
                    // falls through
                case SyntaxKind::EndOfFileToken:
                    // Done
                    goto loop;
                case SyntaxKind::WhitespaceTrivia:
                    if (state == JSDocState::SavingComments || state == JSDocState::SavingBackticks) {
                        pushComment(scanner.getTokenText());
                    }
                    else {
                        auto whitespace = scanner.getTokenText();
                        // if the whitespace crosses the margin, take only the whitespace that passes the margin
                        if (margin != -1 && indent + whitespace.size() > margin) {
                            comments.push_back(whitespace.substr(margin - indent));
                        }
                        indent += whitespace.size();
                    }
                    break;
                case SyntaxKind::OpenBraceToken:
                    state = JSDocState::SavingComments;
                    if (parser->lookAhead<boolean>([&]() { return parser->nextTokenJSDoc() == SyntaxKind::AtToken && scanner.tokenIsIdentifierOrKeyword(parser->nextTokenJSDoc()) && scanner.getTokenText() == S("link");})) 
                    {
                        pushComment(scanner.getTokenText());
                        parser->nextTokenJSDoc();
                        pushComment(scanner.getTokenText());
                        parser->nextTokenJSDoc();
                    }
                    pushComment(scanner.getTokenText());
                    break;
                case SyntaxKind::BacktickToken:
                    if (state == JSDocState::SavingBackticks) {
                        state = JSDocState::SavingComments;
                    }
                    else {
                        state = JSDocState::SavingBackticks;
                    }
                    pushComment(scanner.getTokenText());
                    break;
                case SyntaxKind::AsteriskToken:
                    if (state == JSDocState::BeginningOfLine) {
                        // leading asterisks start recording on the *next* (non-whitespace) token
                        state = JSDocState::SawAsterisk;
                        indent += 1;
                        break;
                    }
                    // record the *.as<a>() comment
                    // falls through
                default:
                    if (state != JSDocState::SavingBackticks) {
                        state = JSDocState::SavingComments; // leading identifiers start recording.as<well>()
                    }
                    pushComment(scanner.getTokenText());
                    break;
            }
            previousWhitespace = parser->token() == SyntaxKind::WhitespaceTrivia;
            tok = parser->nextTokenJSDoc();
        }
        loop:

        removeLeadingNewlines(comments);
        removeTrailingWhitespace(comments);
        return comments.size() == 0 ? string() : join(comments);
    }

    auto parseUnknownTag(number start, Identifier tagName, number indent, string indentText) -> JSDocUnknownTag {
        auto end = parser->getNodePos();
        return parser->finishNode(parser->factory.createJSDocUnknownTag(tagName, parseTrailingTagComments(start, end, indent, indentText)), start, end);
    }

    auto addTag(JSDocTag tag) -> void {
        if (!tag) {
            return;
        }
        if (!tags) {
            tags = NodeArray<JSDocTag>({tag});
            tagsPos = tag->pos;
        }
        else {
            tags.push_back(tag);
        }
        tagsEnd = tag->_end;
    }

    auto tryParseTypeExpression() -> JSDocTypeExpression {
        skipWhitespaceOrAsterisk();
        return parser->token() == SyntaxKind::OpenBraceToken ? parser->parseJSDocTypeExpression() : undefined;
    }

    auto parseBracketNameInPropertyAndParamTag() -> EntityNameWIthBracketed {
        // Looking for something like '[foo]', 'foo', '[foo.bar]' or 'foo.bar'
        auto isBracketed = parseOptionalJsdoc(SyntaxKind::OpenBracketToken);
        if (isBracketed) {
            skipWhitespace();
        }
        // a markdown-quoted name: `arg` is not legal jsdoc, but occurs in the wild
        auto isBackquoted = parseOptionalJsdoc(SyntaxKind::BacktickToken);
        auto name = parseJSDocEntityName();
        if (isBackquoted) {
            parser->parseExpectedTokenJSDoc(SyntaxKind::BacktickToken);
        }
        if (isBracketed) {
            skipWhitespace();
            // May have an optional default, e.g. '[foo = 42]'
            if (parser->parseOptionalToken(SyntaxKind::EqualsToken)) {
                parser->parseExpression();
            }

            parser->parseExpected(SyntaxKind::CloseBracketToken);
        }

        return { name, isBracketed };
    }

    auto isObjectOrObjectArrayTypeReference(TypeNode node) -> boolean {
        switch (node->kind) {
            case SyntaxKind::ObjectKeyword:
                return true;
            case SyntaxKind::ArrayType:
                return isObjectOrObjectArrayTypeReference(node.as<ArrayTypeNode>()->elementType);
            default:
                return parser->isTypeReferenceNode(node) && ts::isIdentifier(node->typeName) && node->typeName->escapedText == S("Object") && !node->typeArguments;
        }
    }

    auto parseParameterOrPropertyTag(number start, Identifier tagName, PropertyLikeParse target, number indent) -> Node {
        auto typeExpression = tryParseTypeExpression();
        auto isNameFirst = !typeExpression;
        skipWhitespaceOrAsterisk();

        auto res = parseBracketNameInPropertyAndParamTag();
        name = res.name;
        isBracketed = res.isBracketed;
        auto indentText = skipWhitespaceOrAsterisk();

        if (isNameFirst) {
            typeExpression = tryParseTypeExpression();
        }

        auto comment = parseTrailingTagComments(start, getNodePos(), indent, indentText);

        auto nestedTypeLiteral = target != PropertyLikeParse::CallbackParameter && parseNestedTypeLiteral(typeExpression, name, target, indent);
        if (nestedTypeLiteral) {
            typeExpression = nestedTypeLiteral;
            isNameFirst = true;
        }
        auto result = target == PropertyLikeParse::Property
            ? parser->factory.createJSDocPropertyTag(tagName, name, isBracketed, typeExpression, isNameFirst, comment)
            : parser->factory.createJSDocParameterTag(tagName, name, isBracketed, typeExpression, isNameFirst, comment);
        return parser->finishNode(result, start);
    }

    auto parseNestedTypeLiteral(JSDocTypeExpression typeExpression, EntityName name, PropertyLikeParse target, number indent) {
        if (typeExpression && isObjectOrObjectArrayTypeReference(typeExpression->type)) {
            auto pos = getNodePos();
            Node child;
            std::vector<JSDocPropertyLikeTag> children;
            while (child = parser->tryParse([&]() { return parseChildParameterOrPropertyTag(target, indent, name); })) {
                if (child->kind == SyntaxKind::JSDocParameterTag || child->kind == SyntaxKind::JSDocPropertyTag) {
                    children = append(children, child);
                }
            }
            if (!children.empty()) {
                auto literal = parser->finishNode(parser->factory.createJSDocTypeLiteral(children, typeExpression->type->kind == SyntaxKind::ArrayType), pos);
                return parser->finishNode(parser->factory.createJSDocTypeExpression(literal), pos);
            }
        }
    }

    auto parseReturnTag(number start, Identifier tagName, number indent, string indentText) -> JSDocReturnTag {
        if (some(tags, isJSDocReturnTag)) {
            parseErrorAt(tagName->pos, scanner.getTokenPos(), Diagnostics::_0_tag_already_specified, tagName->escapedText);
        }

        auto typeExpression = tryParseTypeExpression();
        auto end = getNodePos();
        return parser->finishNode(parser->factory.createJSDocReturnTag(tagName, typeExpression, parseTrailingTagComments(start, end, indent, indentText)), start, end);
    }

    auto parseTypeTag(number start, Identifier tagName, number indent, string indentText) -> JSDocTypeTag {
        if (some(tags, isJSDocTypeTag)) {
            parseErrorAt(tagName->pos, scanner.getTokenPos(), Diagnostics::_0_tag_already_specified, tagName->escapedText);
        }

        auto typeExpression = parseJSDocTypeExpression(/*mayOmitBraces*/ true);
        auto end = getNodePos();
        auto comments = indent != undefined && indentText != undefined ? parseTrailingTagComments(start, end, indent, indentText) : undefined;
        return parser->finishNode(parser->factory.createJSDocTypeTag(tagName, typeExpression, comments), start, end);
    }

    auto parseSeeTag(number start, Identifier tagName, number indent, string indentText) -> JSDocSeeTag {
        auto nameExpression = parseJSDocNameReference();
        auto end = getNodePos();
        auto comments = indent != undefined && indentText != undefined ? parseTrailingTagComments(start, end, indent, indentText) : undefined;
        return parser->finishNode(parser->factory.createJSDocSeeTag(tagName, nameExpression, comments), start, end);
    }

    auto parseAuthorTag(number start, Identifier tagName, number indent, string indentText) -> JSDocAuthorTag {
        auto comments = parseAuthorNameAndEmail() + (parseTrailingTagComments(start, end, indent, indentText) || string());
        return parser->finishNode(parser->factory.createJSDocAuthorTag(tagName, comments || undefined), start);
    }

    auto parseAuthorNameAndEmail() -> string {
        auto std::vector<string> comments;
        auto inEmail = false;
        auto token = scanner.getToken();
        while (token != SyntaxKind::EndOfFileToken && token != SyntaxKind::NewLineTrivia) {
            if (token == SyntaxKind::LessThanToken) {
                inEmail = true;
            }
            else if (token == SyntaxKind::AtToken && !inEmail) {
                break;
            }
            else if (token == SyntaxKind::GreaterThanToken && inEmail) {
                comments.push_back(scanner.getTokenText());
                scanner.setTextPos(scanner.getTokenPos() + 1);
                break;
            }
            comments.push_back(scanner.getTokenText());
            token = nextTokenJSDoc();
        }

        return join(comments);
    }

    auto parseImplementsTag(number start, Identifier tagName, number margin, string indentText) -> JSDocImplementsTag {
        auto className = parseExpressionWithTypeArgumentsForAugments();
        auto end = getNodePos();
        return parser->finishNode(parser->factory.createJSDocImplementsTag(tagName, className, parseTrailingTagComments(start, end, margin, indentText)), start, end);
    }

    auto parseAugmentsTag(number start, Identifier tagName, number margin, string indentText) -> JSDocAugmentsTag {
        auto className = parseExpressionWithTypeArgumentsForAugments();
        auto end = getNodePos();
        return parser->finishNode(parser->factory.createJSDocAugmentsTag(tagName, className, parseTrailingTagComments(start, end, margin, indentText)), start, end);
    }

    auto parseExpressionWithTypeArgumentsForAugments() -> ExpressionWithTypeArguments {
        auto usedBrace = parseOptional(SyntaxKind::OpenBraceToken);
        auto pos = getNodePos();
        auto expression = parsePropertyAccessEntityNameExpression();
        auto typeArguments = tryParseTypeArguments();
        auto node = parser->factory.createExpressionWithTypeArguments(expression, typeArguments);
        auto res = parser->finishNode(node, pos);
        if (usedBrace) {
            parseExpected(SyntaxKind::CloseBraceToken);
        }
        return res;
    }

    auto parsePropertyAccessEntityNameExpression() -> Node {
        auto pos = getNodePos();
        Node node = parseJSDocIdentifierName();
        while (parseOptional(SyntaxKind::DotToken)) {
            auto name = parseJSDocIdentifierName();
            node = parser->finishNode(parser->factory.createPropertyAccessExpression(node, name), pos).as<PropertyAccessEntityNameExpression>();
        }
        return node;
    }

    auto parseSimpleTag(number start, std::function<JSDocTag(Identifier, string)> createTag, Identifier tagName, number margin, string indentText) -> JSDocTag {
        auto end = getNodePos();
        return parser->finishNode(createTag(tagName, parseTrailingTagComments(start, end, margin, indentText)), start, end);
    }

    auto parseThisTag(number start, Identifier tagName, number margin, string indentText) -> JSDocThisTag {
        auto typeExpression = parseJSDocTypeExpression(/*mayOmitBraces*/ true);
        skipWhitespace();
        auto end = getNodePos();
        return parser->finishNode(parser->factory.createJSDocThisTag(tagName, typeExpression, parseTrailingTagComments(start, end, margin, indentText)), start, end);
    }

    auto parseEnumTag(number start, Identifier tagName, number margin, string indentText) -> JSDocEnumTag {
        auto typeExpression = parseJSDocTypeExpression(/*mayOmitBraces*/ true);
        skipWhitespace();
        auto end = getNodePos();
        return parser->finishNode(parser->factory.createJSDocEnumTag(tagName, typeExpression, parseTrailingTagComments(start, end, margin, indentText)), start, end);
    }

    auto parseTypedefTag(number start, Identifier tagName, number indent, string indentText) -> JSDocTypedefTag {
        auto typeExpression = tryParseTypeExpression();
        skipWhitespaceOrAsterisk();

        auto fullName = parseJSDocTypeNameWithNamespace();
        skipWhitespace();
        auto comment = parseTagComments(indent);

        number end;
        if (!typeExpression || isObjectOrObjectArrayTypeReference(typeExpression.type)) {
            Node child;
            JSDocTypeTag childTypeTag;
            std::vector<JSDocPropertyTag> jsDocPropertyTags;
            auto hasChildren = false;
            while (child = tryParse([&]() { return parseChildPropertyTag(indent); })) {
                hasChildren = true;
                if (child->kind == SyntaxKind::JSDocTypeTag) {
                    if (!!childTypeTag) {
                        parseErrorAtCurrentToken(Diagnostics::A_JSDoc_typedef_comment_may_not_contain_multiple_type_tags);
                        auto lastError = lastOrUndefined(parseDiagnostics);
                        if (lastError) {
                            addRelatedInfo(
                                lastError,
                                createDetachedDiagnostic(fileName, 0, 0, Diagnostics::The_tag_was_first_specified_here)
                            );
                        }
                        break;
                    }
                    else {
                        childTypeTag = child;
                    }
                }
                else {
                    jsDocPropertyTags = append(jsDocPropertyTags, child);
                }
            }
            if (hasChildren) {
                auto isArrayType = !!typeExpression && typeExpression->type->kind == SyntaxKind::ArrayType;
                auto jsdocTypeLiteral = parser->factory.createJSDocTypeLiteral(jsDocPropertyTags, isArrayType);
                typeExpression = !!childTypeTag && childTypeTag->typeExpression && !isObjectOrObjectArrayTypeReference(childTypeTag->typeExpression->type) ?
                    childTypeTag->typeExpression :
                    parser->finishNode(jsdocTypeLiteral, start);
                end = typeExpression->_end;
            }
        }

        // Only include the characters between the name end and the next token if a comment was actually parsed out - otherwise it's just whitespace
        end = end || !comment.empty() ?
            getNodePos() :
            (fullName ?? typeExpression ?? tagName)->_end;

        if (!comment) {
            comment = parseTrailingTagComments(start, end, indent, indentText);
        }

        auto typedefTag = parser->factory.createJSDocTypedefTag(tagName, typeExpression, fullName, comment);
        return parser->finishNode(typedefTag, start, end);
    }

    auto parseJSDocTypeNameWithNamespace(boolean nested) {
        auto pos = scanner.getTokenPos();
        if (!scanner.tokenIsIdentifierOrKeyword(token())) {
            return undefined;
        }
        auto typeNameOrNamespaceName = parseJSDocIdentifierName();
        if (parseOptional(SyntaxKind::DotToken)) {
            auto body = parseJSDocTypeNameWithNamespace(/*nested*/ true);
            auto jsDocNamespaceNode = parser->factory.createModuleDeclaration(
                /*decorators*/ undefined,
                /*modifiers*/ undefined,
                typeNameOrNamespaceName,
                body,
                nested ? NodeFlags::NestedNamespace : undefined
            ).as<JSDocNamespaceDeclaration>();
            return parser->finishNode(jsDocNamespaceNode, pos);
        }

        if (nested) {
            typeNameOrNamespaceName.isInJSDocNamespace = true;
        }
        return typeNameOrNamespaceName;
    }


    auto parseCallbackTagParameters(number indent) {
        auto pos = getNodePos();
        auto JSDocParameterTag child | false;
        auto parameters;
        while (child = tryParse(() => parseChildParameterOrPropertyTag(PropertyLikeParse::CallbackParameter, indent).as<JSDocParameterTag>())) {
            parameters = append(parameters, child);
        }
        return createNodeArray(parameters || [], pos);
    }

    auto parseCallbackTag(number start, Identifier tagName, number indent, string indentText) -> JSDocCallbackTag {
        auto fullName = parseJSDocTypeNameWithNamespace();
        skipWhitespace();
        auto comment = parseTagComments(indent);
        auto parameters = parseCallbackTagParameters(indent);
        auto returnTag = tryParse(() => {
            if (parseOptionalJsdoc(SyntaxKind::AtToken)) {
                auto tag = parseTag(indent);
                if (tag && tag->kind == SyntaxKind::JSDocReturnTag) {
                    return tag.as<JSDocReturnTag>();
                }
            }
        });
        auto typeExpression = parser->finishNode(parser->factory.createJSDocSignature(/*typeParameters*/ undefined, parameters, returnTag), start);
        auto end = getNodePos();
        if (!comment) {
            comment = parseTrailingTagComments(start, end, indent, indentText);
        }
        return parser->finishNode(parser->factory.createJSDocCallbackTag(tagName, typeExpression, fullName, comment), start, end);
    }

    auto escapedTextsEqual(EntityName a, EntityName b) -> boolean {
        while (!ts.isIdentifier(a) || !ts.isIdentifier(b)) {
            if (!ts.isIdentifier(a) && !ts.isIdentifier(b) && a.right->escapedText == b.right->escapedText) {
                a = a.left;
                b = b.left;
            }
            else {
                return false;
            }
        }
        return a->escapedText == b->escapedText;
    }

    auto parseChildPropertyTag(number indent) -> JSDocTypeTag {
        return parseChildParameterOrPropertyTag(PropertyLikeParse::Property, indent).as<JSDocTypeTag>();
    }

    auto parseChildParameterOrPropertyTag(PropertyLikeParse target, number indent, EntityName name) -> Node {
        auto canParseTag = true;
        auto seenAsterisk = false;
        while (true) {
            switch (nextTokenJSDoc()) {
                case SyntaxKind::AtToken:
                    if (canParseTag) {
                        auto child = tryParseChildTag(target, indent);
                        if (child && (child->kind == SyntaxKind::JSDocParameterTag || child->kind == SyntaxKind::JSDocPropertyTag) &&
                            target != PropertyLikeParse::CallbackParameter &&
                            name && (ts::isIdentifier(child.name) || !escapedTextsEqual(name, child.name.left))) {
                            return false;
                        }
                        return child;
                    }
                    seenAsterisk = false;
                    break;
                case SyntaxKind::NewLineTrivia:
                    canParseTag = true;
                    seenAsterisk = false;
                    break;
                case SyntaxKind::AsteriskToken:
                    if (seenAsterisk) {
                        canParseTag = false;
                    }
                    seenAsterisk = true;
                    break;
                case SyntaxKind::Identifier:
                    canParseTag = false;
                    break;
                case SyntaxKind::EndOfFileToken:
                    return false;
            }
        }
    }

    auto tryParseChildTag(PropertyLikeParse target, number indent) -> Node {
        Debug::_assert(token() == SyntaxKind::AtToken);
        auto start = scanner.getStartPos();
        nextTokenJSDoc();

        auto tagName = parseJSDocIdentifierName();
        skipWhitespace();
        PropertyLikeParse t;
        switch (tagName->escapedText) {
            case "type":
                return target == PropertyLikeParse::Property && parseTypeTag(start, tagName);
            case "prop":
            case "property":
                t = PropertyLikeParse::Property;
                break;
            case "arg":
            case "argument":
            case "param":
                t = PropertyLikeParse::Parameter | PropertyLikeParse::CallbackParameter;
                break;
            default:
                return false;
        }
        if (!(target & t)) {
            return false;
        }
        return parseParameterOrPropertyTag(start, tagName, target, indent);
    }

    auto parseTemplateTagTypeParameter() {
        auto typeParameterPos = getNodePos();
        auto name = parseJSDocIdentifierName(Diagnostics::Unexpected_token_A_type_parameter_name_was_expected_without_curly_braces);
        if (nodeIsMissing(name)) {
            return undefined;
        }
        return parser->finishNode(parser->factory.createTypeParameterDeclaration(name, /*constraint*/ undefined, /*defaultType*/ undefined), typeParameterPos);
    }

    auto parseTemplateTagTypeParameters() {
        auto pos = getNodePos();
        auto typeParameters = [];
        do {
            skipWhitespace();
            auto node = parseTemplateTagTypeParameter();
            if (node != undefined) {
                typeParameters.push(node);
            }
            skipWhitespaceOrAsterisk();
        } while (parseOptionalJsdoc(SyntaxKind::CommaToken));
        return createNodeArray(typeParameters, pos);
    }

    auto parseTemplateTag(number start, Identifier tagName, number indent, string indentText) -> JSDocTemplateTag {
        // The template tag looks like one of the following:
        //   @template T,U,V
        //   @template {Constraint} T
        //
        // According to the [closure docs](https://github.com/google/closure-compiler/wiki/Generic-Types#multiple-bounded-template-types) ->
        //   > Multiple bounded generics cannot be declared on the same line. For the sake of clarity, if multiple templates share the same
        //   > type bound they must be declared on separate lines.
        //
        // Determine TODO whether we should enforce this in the checker.
        // Consider TODO moving the `constraint` to the first type parameter.as<we>() could then remove `getEffectiveConstraintOfTypeParameter`.
        // Consider TODO only parsing a single type parameter if there is a constraint.
        auto constraint = token() == SyntaxKind::OpenBraceToken ? parseJSDocTypeExpression() : undefined;
        auto typeParameters = parseTemplateTagTypeParameters();
        auto end = getNodePos();
        return parser->finishNode(parser->factory.createJSDocTemplateTag(tagName, constraint, typeParameters, parseTrailingTagComments(start, end, indent, indentText)), start, end);
    }

    auto parseOptionalJsdoc(SyntaxKind t) -> boolean {
        if (token() == t) {
            nextTokenJSDoc();
            return true;
        }
        return false;
    }

    auto parseJSDocEntityName() -> EntityName {
        EntityName entity = parseJSDocIdentifierName();
        if (parseOptional(SyntaxKind::OpenBracketToken)) {
            parseExpected(SyntaxKind::CloseBracketToken);
            // Note that y[] is accepted.as<an>() entity name, but the postfix brackets are not saved for checking.
            // Technically usejsdoc.org requires them for specifying a property of a type equivalent to Array<{ ... x}>
            // but it's not worth it to enforce that restriction.
        }
        while (parseOptional(SyntaxKind::DotToken)) {
            auto name = parseJSDocIdentifierName();
            if (parseOptional(SyntaxKind::OpenBracketToken)) {
                parseExpected(SyntaxKind::CloseBracketToken);
            }
            entity = createQualifiedName(entity, name);
        }
        return entity;
    }

    auto parseJSDocIdentifierName(DiagnosticMessage message) -> Identifier {
        if (!scanner.tokenIsIdentifierOrKeyword(token())) {
            return createMissingNode<Identifier>(SyntaxKind::Identifier, /*reportAtCurrentPosition*/ !message, message || Diagnostics::Identifier_expected);
        }

        identifierCount++;
        auto pos = scanner.getTokenPos();
        auto end = scanner.getTextPos();
        auto originalKeywordKind = token();
        auto text = internIdentifier(scanner.getTokenValue());
        auto result = parser->finishNode(parser->factory.createIdentifier(text, /*typeArguments*/ undefined, originalKeywordKind), pos, end);
        nextTokenJSDoc();
        return result;
    }
};

// End JSDoc namespace
