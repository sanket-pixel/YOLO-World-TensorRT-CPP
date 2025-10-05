#include "text_preprocessor.h"
#include <sstream>

namespace PromptDetector {

TextPreprocessor::TextPreprocessor(const std::string& vocab_path) {
    const int start_token = 49406;
    const int end_token = 49407;
    const size_t max_tokens = 6;
    const size_t text_count = 2;
    tokenizer_ = std::make_unique<Tokenizer>(vocab_path, start_token, end_token, max_tokens, text_count);
}

void TextPreprocessor::Process(const std::string& prompt, std::vector<int32_t>& token_ids_output, std::vector<int32_t>& attention_mask_output) {
    auto token_map = tokenizer_->tokenize(prompt);
    const auto& tokens_vec = token_map.at("text_prompt_tokens");
    const auto& mask_vec = token_map.at("text_prompt_padding_mask");
    token_ids_output.clear();
    attention_mask_output.clear();
    for (const auto& vec : tokens_vec) {
        token_ids_output.insert(token_ids_output.end(), vec.begin(), vec.end());
    }
    for (const auto& vec : mask_vec) {
        attention_mask_output.insert(attention_mask_output.end(), vec.begin(), vec.end());
    }
}

const std::vector<std::string>& TextPreprocessor::GetClassNames() const {
    return class_names_;
}

}

