#ifndef PROMPT_DETECTOR_TOKENIZER_H
#define PROMPT_DETECTOR_TOKENIZER_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace PromptDetector {

class Tokenizer {
public:
  Tokenizer(const std::string& vocab_path, int start_token = 49406, int end_token = 49407,
            std::size_t max_words_per_text = 6, std::size_t batch_size = 2);

  std::map<std::string, std::vector<std::vector<int>>> tokenize(const std::string& text);

private:
  std::unordered_map<std::string, int> vocabulary_;
  int start_id_{49406};
  int end_id_{49407};
  std::size_t max_words_per_text_{6};
  std::size_t text_count_{2};

  void LoadVocab(const std::string& vocabulary_path);
};

}

#endif // PROMPT_DETECTOR_TOKENIZER_H
