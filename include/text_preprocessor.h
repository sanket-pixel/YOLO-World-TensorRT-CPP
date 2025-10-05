#ifndef PROMPT_DETECTOR_TEXT_PREPROCESSOR_H
#define PROMPT_DETECTOR_TEXT_PREPROCESSOR_H

#include "tokenizer.h"
#include <string>
#include <vector>
#include <memory>

namespace PromptDetector {

class TextPreprocessor {
public:
  explicit TextPreprocessor(const std::string& vocab_path);

  void Process(const std::string& prompt, std::vector<int32_t>& token_ids_output, std::vector<int32_t>& attention_mask_output);
  const std::vector<std::string>& GetClassNames() const;

private:
  std::vector<std::string> ParsePrompt(const std::string& prompt);

  std::unique_ptr<Tokenizer> tokenizer_;
  std::vector<std::string> class_names_;
};

}

#endif // PROMPT_DETECTOR_TEXT_PREPROCESSOR_H

