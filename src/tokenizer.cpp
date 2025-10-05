#include "tokenizer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

namespace PromptDetector {
Tokenizer::Tokenizer(const std::string& vocab_path, int start_token, int end_token, std::size_t max_words_per_text,
                     std::size_t text_count)
    : start_id_(start_token), end_id_(end_token), max_words_per_text_(max_words_per_text), text_count_(text_count) {
  LoadVocab(vocab_path);
}

void Tokenizer::LoadVocab(const std::string& vocab_file_path) {
  std::ifstream vocab_file(vocab_file_path);
  if (!vocab_file.is_open()) {
    return;
  }
  nlohmann::json vocab_json_data;
  vocab_file >> vocab_json_data;
  for (auto json_it = vocab_json_data.begin(); json_it != vocab_json_data.end(); ++json_it) {
    const std::string& token_string = json_it.key();
    int token_id = json_it.value().get<int>();
    vocabulary_[token_string] = token_id;
  }
}

std::map<std::string, std::vector<std::vector<int>>> Tokenizer::tokenize(const std::string& input_text) {
  std::vector<std::vector<int>> token_ids_output(text_count_);
  std::vector<std::vector<int>> attention_mask_output(text_count_);
  token_ids_output[0].push_back(start_id_);
  std::istringstream text_stream(input_text);
  std::string current_word;
  while (text_stream >> current_word) {
    auto vocabulary_iterator = vocabulary_.find(current_word + "</w>");
    if (vocabulary_iterator != vocabulary_.end()) {
      token_ids_output[0].push_back(vocabulary_iterator->second);
    } else {
      std::cerr << "Warning: '" << current_word << "' not found in vocab.\n";
    }
  }
  token_ids_output[0].push_back(end_id_);
  attention_mask_output[0] = std::vector<int>(token_ids_output[0].size(), 1);
  while (token_ids_output[0].size() < max_words_per_text_) {
    token_ids_output[0].push_back(end_id_);
    attention_mask_output[0].push_back(0);
  }
  token_ids_output[1] = {start_id_, end_id_};
  attention_mask_output[1] = {1, 1};
  while (token_ids_output[1].size() < max_words_per_text_) {
    token_ids_output[1].push_back(end_id_);
    attention_mask_output[1].push_back(0);
  }
  return {{"text_prompt_tokens", token_ids_output}, {"text_prompt_padding_mask", attention_mask_output}};
}
}
