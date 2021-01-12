
// Generated from c:\dev\TypeScriptCompiler\tsc\tsc-grammar\TypeScriptParserANTLR.g4 by ANTLR 4.8


#include "TypeScriptParserANTLRListener.h"
#include "TypeScriptParserANTLRVisitor.h"

#include "TypeScriptParserANTLR.h"


using namespace antlrcpp;
using namespace typescript;
using namespace antlr4;

TypeScriptParserANTLR::TypeScriptParserANTLR(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

TypeScriptParserANTLR::~TypeScriptParserANTLR() {
  delete _interpreter;
}

std::string TypeScriptParserANTLR::getGrammarFileName() const {
  return "TypeScriptParserANTLR.g4";
}

const std::vector<std::string>& TypeScriptParserANTLR::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& TypeScriptParserANTLR::getVocabulary() const {
  return _vocabulary;
}


//----------------- MainContext ------------------------------------------------------------------

TypeScriptParserANTLR::MainContext::MainContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* TypeScriptParserANTLR::MainContext::EOF() {
  return getToken(TypeScriptParserANTLR::EOF, 0);
}


size_t TypeScriptParserANTLR::MainContext::getRuleIndex() const {
  return TypeScriptParserANTLR::RuleMain;
}

void TypeScriptParserANTLR::MainContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TypeScriptParserANTLRListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMain(this);
}

void TypeScriptParserANTLR::MainContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<TypeScriptParserANTLRListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMain(this);
}


antlrcpp::Any TypeScriptParserANTLR::MainContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<TypeScriptParserANTLRVisitor*>(visitor))
    return parserVisitor->visitMain(this);
  else
    return visitor->visitChildren(this);
}

TypeScriptParserANTLR::MainContext* TypeScriptParserANTLR::main() {
  MainContext *_localctx = _tracker.createInstance<MainContext>(_ctx, getState());
  enterRule(_localctx, 0, TypeScriptParserANTLR::RuleMain);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(2);
    match(TypeScriptParserANTLR::EOF);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

// Static vars and initialization.
std::vector<dfa::DFA> TypeScriptParserANTLR::_decisionToDFA;
atn::PredictionContextCache TypeScriptParserANTLR::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN TypeScriptParserANTLR::_atn;
std::vector<uint16_t> TypeScriptParserANTLR::_serializedATN;

std::vector<std::string> TypeScriptParserANTLR::_ruleNames = {
  "main"
};

std::vector<std::string> TypeScriptParserANTLR::_literalNames = {
};

std::vector<std::string> TypeScriptParserANTLR::_symbolicNames = {
};

dfa::Vocabulary TypeScriptParserANTLR::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> TypeScriptParserANTLR::_tokenNames;

TypeScriptParserANTLR::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x2, 0x7, 0x4, 0x2, 0x9, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x2, 
    0x2, 0x3, 0x2, 0x2, 0x2, 0x2, 0x5, 0x2, 0x4, 0x3, 0x2, 0x2, 0x2, 0x4, 
    0x5, 0x7, 0x2, 0x2, 0x3, 0x5, 0x3, 0x3, 0x2, 0x2, 0x2, 0x2, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

TypeScriptParserANTLR::Initializer TypeScriptParserANTLR::_init;
