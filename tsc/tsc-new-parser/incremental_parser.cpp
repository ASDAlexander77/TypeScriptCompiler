#include "core.h"
#include "node_factory.h"
#include "node_test.h"
#include "parser.h"
#include "utilities.h"

// namespace ts
// {
// namespace IncrementalParser
// {
// struct IncrementalParser
// {
//     auto updateSourceFile(SourceFile sourceFile, string newText, TextChangeRange textChangeRange, boolean aggressiveChecks) -> SourceFile
//     {
//         aggressiveChecks = aggressiveChecks || Debug::shouldAssert(AssertionLevel::Aggressive);

//         checkChangeRange(sourceFile, newText, textChangeRange, aggressiveChecks);
//         if (textChangeRangeIsUnchanged(textChangeRange))
//         {
//             // if the text didn't change, then we can just return our current source file as-is.
//             return sourceFile;
//         }

//         if (sourceFile->statements.size() == 0)
//         {
//             // If we don't have any statements in the current source file, then there's no real
//             // way to incrementally parse.  So just do a full parse instead.
//             return Parser::parseSourceFile(sourceFile->fileName, newText, sourceFile->languageVersion, undefined, /*setParentNodes*/
//             true,
//                                            sourceFile->scriptKind);
//         }

//         // Make sure we're not trying to incrementally update a source file more than once.  Once
//         // we do an update the original source file is considered unusable from that point onwards.
//         //
//         // This is because we do incremental parsing in-place.  i.e. we take nodes from the old
//         // tree and give them new positions and parents.  From that point on, trusting the old
//         // tree at all is not possible.as<far>() too much of it may violate invariants.
//         auto incrementalSourceFile = (IncrementalNode)sourceFile->as<Node>();
//         Debug::_assert(!incrementalSourceFile.hasBeenIncrementallyParsed);
//         incrementalSourceFile.hasBeenIncrementallyParsed = true;
//         Parser::fixupParentReferences(incrementalSourceFile);
//         auto oldText = sourceFile->text;
//         auto syntaxCursor = createSyntaxCursor(sourceFile);

//         // Make the actual change larger so that we know to reparse anything whose lookahead
//         // might have intersected the change.
//         auto changeRange = extendToAffectedRange(sourceFile, textChangeRange);
//         checkChangeRange(sourceFile, newText, changeRange, aggressiveChecks);

//         // Ensure that extending the affected range only moved the start of the change range
//         // earlier in the file.
//         Debug::_assert(changeRange.span.start <= textChangeRange.span.start);
//         Debug::_assert(textSpanEnd(changeRange.span) == textSpanEnd(textChangeRange.span));
//         Debug::_assert(textSpanEnd(textChangeRangeNewSpan(changeRange)) == textSpanEnd(textChangeRangeNewSpan(textChangeRange)));

//         // The is the amount the nodes after the edit range need to be adjusted.  It can be
//         // positive (if the edit added characters), negative (if the edit deleted characters)
//         // or zero (if this was a pure overwrite with nothing added/removed).
//         auto delta = textChangeRangeNewSpan(changeRange).size() - changeRange.span.size();

//         // If we added or removed characters during the edit, then we need to go and adjust all
//         // the nodes after the edit.  Those nodes may move forward (if we inserted chars) or they
//         // may move backward (if we deleted chars).
//         //
//         // Doing this helps us out in two ways.  First, it means that any nodes/tokens we want
//         // to reuse are already at the appropriate position in the new text.  That way when we
//         // reuse them, we don't have to figure out if they need to be adjusted.  Second, it makes
//         // it very easy to determine if we can reuse a node->  If the node's position is at where
//         // we are in the text, then we can reuse it.  Otherwise we can't.  If the node's position
//         // is ahead of us, then we'll need to rescan tokens.  If the node's position is behind
//         // us, then we'll need to skip it or crumble it.as<appropriate>()
//         //
//         // We will also adjust the positions of nodes that intersect the change range.as<well>().
//         // By doing this, we ensure that all the positions in the old tree are consistent, not
//         // just the positions of nodes entirely before/after the change range.  By being
//         // consistent, we can then easily map from positions to nodes in the old tree easily.
//         //
//         // Also, mark any syntax elements that intersect the changed span.  We know, up front,
//         // that we cannot reuse these elements.
//         updateTokenPositionsAndMarkElements(incrementalSourceFile, changeRange.span.start, textSpanEnd(changeRange.span),
//                                             textSpanEnd(textChangeRangeNewSpan(changeRange)), delta, oldText, newText, aggressiveChecks);

//         // Now that we've set up our internal incremental state just proceed and parse the
//         // source file in the normal fashion.  When possible the parser will retrieve and
//         // reuse nodes from the old tree.
//         //
//         // passing Note in 'true' for setNodeParents is very important.  When incrementally
//         // parsing, we will be reusing nodes from the old tree, and placing it into new
//         // parents.  If we don't set the parents now, we'll end up with an observably
//         // inconsistent tree.  Setting the parents on the new tree should be very fast.  We
//         // will immediately bail out of walking any subtrees when we can see that their parents
//         // are already correct.
//         auto result = Parser::parseSourceFile(sourceFile->fileName, newText, sourceFile->languageVersion, syntaxCursor,
//                                               /*setParentNodes*/ true, sourceFile->scriptKind);
//         result.commentDirectives = getNewCommentDirectives(sourceFile->commentDirectives, result.commentDirectives,
//         changeRange.span.start,
//                                                            textSpanEnd(changeRange.span), delta, oldText, newText, aggressiveChecks);
//         return result;
//     }

//     auto getNewCommentDirectives(NodeArray<CommentDirective> oldDirectives, NodeArray<CommentDirective> newDirectives, number
//     changeStart,
//                                  number changeRangeOldEnd, number delta, safe_string oldText, safe_string newText, boolean
//                                  aggressiveChecks)
//         -> NodeArray<CommentDirective>
//     {
//         if (!oldDirectives)
//             return newDirectives;
//         NodeArray<CommentDirective> commentDirectives;
//         auto addedNewlyScannedDirectives = false;

//         auto addNewlyScannedDirectives = [&]() {
//             if (addedNewlyScannedDirectives)
//                 return;
//             addedNewlyScannedDirectives = true;
//             if (!commentDirectives)
//             {
//                 commentDirectives = newDirectives;
//             }
//             else if (!newDirectives.empty())
//             {
//                 for (auto &directive : newDirectives)
//                 {
//                     commentDirectives.push_back(directive);
//                 }
//             }
//         };

//         for (auto &directive : oldDirectives)
//         {
//             auto range = directive.range;
//             auto type = directive.type;
//             // Range before the change
//             if (range.end < changeStart)
//             {
//                 commentDirectives = append(commentDirectives, directive);
//             }
//             else if (range.pos > changeRangeOldEnd)
//             {
//                 addNewlyScannedDirectives();
//                 // Node is entirely past the change range.  We need to move both its pos and
//                 // end, forward or backward appropriately.
//                 CommentDirective updatedDirective = {{range.pos + delta, range.end + delta}, type};
//                 commentDirectives = append(commentDirectives, updatedDirective);
//                 if (aggressiveChecks)
//                 {
//                     Debug::_assert(oldText.substring(range.pos, range.end) ==
//                                    newText.substring(updatedDirective.range.pos, updatedDirective.range.end));
//                 }
//             }
//             // Ignore ranges that fall in change range
//         }
//         addNewlyScannedDirectives();
//         return commentDirectives;
//     }

//     auto moveElementEntirelyPastChangeRange(IncrementalElement element, boolean isArray, number delta, string oldText, string newText,
//                                             boolean aggressiveChecks)
//     {
//         auto visitNode =
//             [&](IncrementalNode node) {
//                 auto text = string();
//                 if (aggressiveChecks && shouldCheckNode(node))
//                 {
//                     text = safe_string(oldText).substring(node->pos, node->_end);
//                 }

//                 // Ditch any existing LS children we may have created.  This way we can avoid
//                 // moving them forward.
//                 if (node->_children)
//                 {
//                     node->_children = undefined;
//                 }

//                 setTextRangePosEnd(node, node->pos + delta, node->_end + delta);

//                 if (aggressiveChecks && shouldCheckNode(node))
//                 {
//                     Debug::_assert(text == safe_string(newText).substring(node->pos, node->_end));
//                 }

//                 forEachChild(node, visitNode, visitArray);
//                 if (hasJSDocNodes(node))
//                 {
//                     for (auto jsDocComment : node->jsDoc)
//                     {
//                         visitNode((IncrementalNode)jsDocComment.as<Node>());
//                     }
//                 }
//                 checkNodePositions(node, aggressiveChecks);
//             }

//         auto visitArray =
//             [&](IncrementalNodeArray array) {
//                 array._children = undefined;
//                 setTextRangePosEnd(array, array->pos + delta, array.end + delta);

//                 for (auto node : array)
//                 {
//                     visitNode(node);
//                 }
//             }

//         if (isArray)
//         {
//             visitArray(element.as<IncrementalNodeArray>());
//         }
//         else
//         {
//             visitNode(element.as<IncrementalNode>());
//         }

//         return;
//     }

//     auto shouldCheckNode(Node node)
//     {
//         switch (node->kind)
//         {
//         case SyntaxKind::StringLiteral:
//         case SyntaxKind::NumericLiteral:
//         case SyntaxKind::Identifier:
//             return true;
//         }

//         return false;
//     }

//     auto adjustIntersectingElement(IncrementalElement element, number changeStart, number changeRangeOldEnd, number changeRangeNewEnd,
//                                    number delta)
//     {
//         Debug::_assert(element.end >= changeStart, S("Adjusting an element that was entirely before the change range"));
//         Debug::_assert(element.pos <= changeRangeOldEnd, S("Adjusting an element that was entirely after the change range"));
//         Debug::_assert(element.pos <= element.end);

//         // We have an element that intersects the change range in some way.  It may have its
//         // start, or its end (or both) in the changed range.  We want to adjust any part
//         // that intersects such that the final tree is in a consistent state.  i.e. all
//         // children have spans within the span of their parent, and all siblings are ordered
//         // properly.

//         // We may need to update both the 'pos' and the 'end' of the element.

//         // If the 'pos' is before the start of the change, then we don't need to touch it.
//         // If it isn't, then the 'pos' must be inside the change.  How we update it will
//         // depend if delta is positive or negative. If delta is positive then we have
//         // something like:
//         //
//         //  -------------------AAA-----------------
//         //  -------------------BBBCCCCCCC-----------------
//         //
//         // In this case, we consider any node that started in the change range to still be
//         // starting at the same position.
//         //
//         // however, if the delta is negative, then we instead have something like this:
//         //
//         //  -------------------XXXYYYYYYY-----------------
//         //  -------------------ZZZ-----------------
//         //
//         // In this case, any element that started in the 'X' range will keep its position.
//         // However any element that started after that will have their pos adjusted to be
//         // at the end of the new range.  i.e. any node that started in the 'Y' range will
//         // be adjusted to have their start at the end of the 'Z' range.
//         //
//         // The element will keep its position if possible.  Or Move backward to the new-end
//         // if it's in the 'Y' range.
//         auto pos = std::min(element.pos, changeRangeNewEnd);

//         // If the 'end' is after the change range, then we always adjust it by the delta
//         // amount.  However, if the end is in the change range, then how we adjust it
//         // will depend on if delta is positive or negative.  If delta is positive then we
//         // have something like:
//         //
//         //  -------------------AAA-----------------
//         //  -------------------BBBCCCCCCC-----------------
//         //
//         // In this case, we consider any node that ended inside the change range to keep its
//         // end position.
//         //
//         // however, if the delta is negative, then we instead have something like this:
//         //
//         //  -------------------XXXYYYYYYY-----------------
//         //  -------------------ZZZ-----------------
//         //
//         // In this case, any element that ended in the 'X' range will keep its position.
//         // However any element that ended after that will have their pos adjusted to be
//         // at the end of the new range.  i.e. any node that ended in the 'Y' range will
//         // be adjusted to have their end at the end of the 'Z' range.
//         auto end = element.end >= changeRangeOldEnd ?
//                                                     // Element ends after the change range.  Always adjust the end pos.
//                        element.end + delta
//                                                     :
//                                                     // Element ends in the change range.  The element will keep its position if
//                        // possible. Or Move backward to the new-end if it's in the 'Y' range.
//                        std::min(element.end, changeRangeNewEnd);

//         Debug::_assert(pos <= end);
//         if (element.parent)
//         {
//             Debug::_assertGreaterThanOrEqual(pos, element.parent.pos);
//             Debug::_assertLessThanOrEqual(end, element.parent.end);
//         }

//         setTextRangePosEnd(element, pos, end);
//     }

//     auto checkNodePositions(Node node, boolean aggressiveChecks)
//     {
//         if (aggressiveChecks)
//         {
//             auto pos = node->pos;
//             auto visitNode = [&](Node child) {
//                 Debug::_assert(child->pos >= pos);
//                 pos = child.end;
//             };
//             if (hasJSDocNodes(node))
//             {
//                 for (auto jsDocComment : node->jsDoc)
//                 {
//                     visitNode(jsDocComment);
//                 }
//             }
//             forEachChild(node, visitNode);
//             Debug::_assert(pos <= node->_end);
//         }
//     }

//     auto updateTokenPositionsAndMarkElements(IncrementalNode sourceFile, number changeStart, number changeRangeOldEnd,
//                                              number changeRangeNewEnd, number delta, string oldText, string newText,
//                                              boolean aggressiveChecks) -> void
//     {
//         auto visitNode = [&](IncrementalNode child) {
//             Debug::_assert(child->pos <= child.end);
//             if (child->pos > changeRangeOldEnd)
//             {
//                 // Node is entirely past the change range.  We need to move both its pos and
//                 // end, forward or backward appropriately.
//                 moveElementEntirelyPastChangeRange(child, /*isArray*/ false, delta, oldText, newText, aggressiveChecks);
//                 return;
//             }

//             // Check if the element intersects the change range.  If it does, then it is not
//             // reusable.  Also, we'll need to recurse to see what constituent portions we may
//             // be able to use.
//             auto fullEnd = child.end;
//             if (fullEnd >= changeStart)
//             {
//                 child.intersectsChange = true;
//                 child._children = undefined;

//                 // Adjust the pos or end (or both) of the intersecting element accordingly.
//                 adjustIntersectingElement(child, changeStart, changeRangeOldEnd, changeRangeNewEnd, delta);
//                 forEachChild(child, visitNode, visitArray);
//                 if (hasJSDocNodes(child))
//                 {
//                     for (auto jsDocComment : child.jsDoc)
//                     {
//                         visitNode((IncrementalNode)jsDocComment.as<Node>());
//                     }
//                 }
//                 checkNodePositions(child, aggressiveChecks);
//                 return;
//             }

//             // Otherwise, the node is entirely before the change range.  No need to do anything with it.
//             Debug::_assert(fullEnd < changeStart);
//         };

//         auto visitArray = [&](IncrementalNodeArray array) {
//             Debug::_assert(array->pos <= array.end);
//             if (array->pos > changeRangeOldEnd)
//             {
//                 // Array is entirely after the change range.  We need to move it, and move any of
//                 // its children.
//                 moveElementEntirelyPastChangeRange(array, /*isArray*/ true, delta, oldText, newText, aggressiveChecks);
//                 return;
//             }

//             // Check if the element intersects the change range.  If it does, then it is not
//             // reusable.  Also, we'll need to recurse to see what constituent portions we may
//             // be able to use.
//             auto fullEnd = array.end;
//             if (fullEnd >= changeStart)
//             {
//                 array.intersectsChange = true;
//                 array._children = undefined;

//                 // Adjust the pos or end (or both) of the intersecting array accordingly.
//                 adjustIntersectingElement(array, changeStart, changeRangeOldEnd, changeRangeNewEnd, delta);
//                 for (auto node : array)
//                 {
//                     visitNode(node);
//                 }
//                 return;
//             }

//             // Otherwise, the array is entirely before the change range.  No need to do anything with it.
//             Debug::_assert(fullEnd < changeStart);
//         };

//         visitNode(sourceFile);
//         return;
//     }

//     auto extendToAffectedRange(SourceFile sourceFile, TextChangeRange changeRange) -> TextChangeRange
//     {
//         // Consider the following code:
//         //      void foo() { /; }
//         //
//         // If the text changes with an insertion of / just before the semicolon then we end up with:
//         //      void foo() { //; }
//         //
//         // If we were to just use the changeRange a is, then we would not rescan the { token
//         // (as it does not intersect the actual original change range).  Because an edit may
//         // change the token touching it, we actually need to look back *at least* one token so
//         // that the prior token sees that change.
//         auto maxLookahead = 1;

//         auto start = changeRange.span.start;

//         // the first iteration aligns us with the change start. subsequent iteration move us to
//         // the left by maxLookahead tokens.  We only need to do this.as<long>().as<we>()'re not at the
//         // start of the tree.
//         for (auto i = 0; start > 0 && i <= maxLookahead; i++)
//         {
//             auto nearestNode = findNearestNodeStartingBeforeOrAtPosition(sourceFile, start);
//             Debug::_assert(nearestnode->pos <= start);
//             auto position = nearestnode->pos;

//             start = Math.max(0, position - 1);
//         }

//         auto finalSpan = createTextSpanFromBounds(start, textSpanEnd(changeRange.span));
//         auto finalLength = changeRange.newLength + (changeRange.span.start - start);

//         return createTextChangeRange(finalSpan, finalLength);
//     }

//     auto findNearestNodeStartingBeforeOrAtPosition(SourceFile sourceFile, number position) -> Node
//     {
//         auto bestResult = sourceFile;
//         Node lastNodeEntirelyBeforePosition;

//         auto getLastDescendant = [&](Node node) -> Node {
//             while (true)
//             {
//                 auto lastChild = getLastChild(node);
//                 if (lastChild)
//                 {
//                     node = lastChild;
//                 }
//                 else
//                 {
//                     return node;
//                 }
//             }
//         };

//         auto visit = [&](Node child) {
//             if (nodeIsMissing(child))
//             {
//                 // Missing nodes are effectively invisible to us.  We never even consider them
//                 // When trying to find the nearest node before us.
//                 return;
//             }

//             // If the child intersects this position, then this node is currently the nearest
//             // node that starts before the position.
//             if (child->pos <= position)
//             {
//                 if (child->pos >= bestResult->pos)
//                 {
//                     // This node starts before the position, and is closer to the position than
//                     // the previous best node we found.  It is now the new best node->
//                     bestResult = child;
//                 }

//                 // Now, the node may overlap the position, or it may end entirely before the
//                 // position.  If it overlaps with the position, then either it, or one of its
//                 // children must be the nearest node before the position.  So we can just
//                 // recurse into this child to see if we can find something better.
//                 if (position < child.end)
//                 {
//                     // The nearest node is either this child, or one of the children inside
//                     // of it.  We've already marked this child.as<the>() best so far.  Recurse
//                     // in case one of the children is better.
//                     forEachChild(child, visit);

//                     // Once we look at the children of this node, then there's no need to
//                     // continue any further.
//                     return true;
//                 }
//                 else
//                 {
//                     Debug::_assert(child.end <= position);
//                     // The child ends entirely before this position.  Say you have the following
//                     // (where $ is the position)
//                     //
//                     //      <complex expr 1> ? <complex expr 2> $ : <...> <...>
//                     //
//                     // We would want to find the nearest preceding node in "complex expr 2".
//                     // To support that, we keep track of this node, and once we're done searching
//                     // for a best node, we recurse down this node to see if we can find a good
//                     // result in it.
//                     //
//                     // This approach allows us to quickly skip over nodes that are entirely
//                     // before the position, while still allowing us to find any nodes in the
//                     // last one that might be what we want.
//                     lastNodeEntirelyBeforePosition = child;
//                 }
//             }
//             else
//             {
//                 Debug::_assert(child->pos > position);
//                 // We're now at a node that is entirely past the position we're searching for.
//                 // This node (and all following nodes) could never contribute to the result,
//                 // so just skip them by returning 'true' here.
//                 return true;
//             }
//         };

//         forEachChild(sourceFile, visit);

//         if (lastNodeEntirelyBeforePosition)
//         {
//             auto lastChildOfLastEntireNodeBeforePosition = getLastDescendant(lastNodeEntirelyBeforePosition);
//             if (lastChildOfLastEntireNodeBeforePosition->pos > bestResult->pos)
//             {
//                 bestResult = lastChildOfLastEntireNodeBeforePosition;
//             }
//         }

//         return bestResult;
//     }

//     static auto checkChangeRange(SourceFile sourceFile, string newText, TextChangeRange textChangeRange, boolean aggressiveChecks)
//     {
//         auto oldText = sourceFile->text;
//         if (textChangeRange)
//         {
//             Debug::_assert((oldText.size() - textChangeRange.span.size() + textChangeRange.newLength) == newText.size());

//             if (aggressiveChecks || Debug::shouldAssert(AssertionLevel::VeryAggressive))
//             {
//                 auto oldTextPrefix = oldText.substr(0, textChangeRange.span.start);
//                 auto newTextPrefix = newText.substr(0, textChangeRange.span.start);
//                 Debug::_assert(oldTextPrefix == newTextPrefix);

//                 auto oldTextSuffix = safe_string(oldText).substring(textSpanEnd(textChangeRange.span), oldText.size());
//                 auto newTextSuffix = safe_string(newText).substring(textSpanEnd(textChangeRangeNewSpan(textChangeRange)),
//                 newText.size()); Debug::_assert(oldTextSuffix == newTextSuffix);
//             }
//         }
//     }

//     auto createSyntaxCursor(SourceFile sourceFile) -> SyntaxCursor
//     {
//         NodeArray<Node> currentArray = sourceFile->statements;
//         auto currentArrayIndex = 0;

//         Debug::_assert(currentArrayIndex < currentArray.size());
//         auto current = currentArray[currentArrayIndex];
//         auto lastQueriedPosition = InvalidPosition::Value;

//         // Finds the highest element in the tree we can find that starts at the provided position.
//         // The element must be a direct child of some node list in the tree.  This way after we
//         // return it, we can easily return its next sibling in the list.
//         auto findHighestListElementThatStartsAtPosition = [&](number position) {
//             // Clear out any cached state about the last node we found.
//             currentArray = undefined;
//             currentArrayIndex = InvalidPosition::Value;
//             current = undefined;

//             std::function<boolean(Node)> visitNode;
//             std::function<boolean(NodeArray<Node>)> visitArray;

//             visitNode = [&](Node node) {
//                 if (position >= node->pos && position < node->_end)
//                 {
//                     // Position was within this node->  Keep searching deeper to find the node->
//                     forEachChild(node, visitNode, visitArray);

//                     // don't proceed any further in the search.
//                     return true;
//                 }

//                 // position wasn't in this node, have to keep searching.
//                 return false;
//             };

//             visitArray = [&](NodeArray<Node> array) -> boolean {
//                 if (position >= array->pos && position < array.end)
//                 {
//                     // position was in this array.  Search through this array to see if we find a
//                     // viable element.
//                     for (auto i = 0; i < array.size(); i++)
//                     {
//                         auto child = array[i];
//                         if (child)
//                         {
//                             if (child->pos == position)
//                             {
//                                 // Found the right node->  We're done.
//                                 currentArray = array;
//                                 currentArrayIndex = i;
//                                 current = child;
//                                 return true;
//                             }
//                             else
//                             {
//                                 if (child->pos < position && position < child.end)
//                                 {
//                                     // Position in somewhere within this child.  Search in it and
//                                     // stop searching in this array.
//                                     forEachChild(child, visitNode, visitArray);
//                                     return true;
//                                 }
//                             }
//                         }
//                     }
//                 }

//                 // position wasn't in this array, have to keep searching.
//                 return false;
//             };

//             // Recurse into the source file to find the highest node at this position.
//             forEachChild(sourceFile, visitNode, visitArray);
//             return;
//         };

//         SyntaxCursor syntaxCursor;
//         syntaxCursor.currentNode = [&](number position) {
//             // Only compute the current node if the position is different than the last time
//             // we were asked.  The parser commonly asks for the node at the same position
//             // twice.  Once to know if can read an appropriate list element at a certain point,
//             // and then to actually read and consume the node->
//             if (position != lastQueriedPosition)
//             {
//                 // Much of the time the parser will need the very next node in the array that
//                 // we just returned a node from.So just simply check for that case and move
//                 // forward in the array instead of searching for the node again.
//                 if (current && current.end == position && currentArrayIndex < (currentArray.size() - 1))
//                 {
//                     currentArrayIndex++;
//                     current = currentArray[currentArrayIndex];
//                 }

//                 // If we don't have a node, or the node we have isn't in the right position,
//                 // then try to find a viable node at the position requested.
//                 if (!current || current->pos != position)
//                 {
//                     findHighestListElementThatStartsAtPosition(position);
//                 }
//             }

//             // Cache this query so that we don't do any extra work if the parser calls back
//             // into us.  this Note is very common.as<the>() parser will make pairs of calls like
//             // 'isListElement -> parseListElement'.  If we were unable to find a node when
//             // called with 'isListElement', we don't want to redo the work when parseListElement
//             // is called immediately after.
//             lastQueriedPosition = position;

//             // Either we don'd have a node, or we have a node at the position being asked for.
//             Debug::_assert(!current || current->pos == position);
//             return current.as<IncrementalNode>();
//         };

//         return syntaxCursor;
//     }
// }

// } // namespace IncrementalParser
// } // namespace ts
