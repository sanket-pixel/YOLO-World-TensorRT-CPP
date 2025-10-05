#include "gtest/gtest.h"
#include "text_preprocessor.h"
#include <vector>
#include <string>

// This test fixture now directly uses the real vocab file.
// The test/CMakeLists.txt is configured to copy the project's 'weights'
// directory into the build directory, so the test can find it.
class TextPreprocessorTest : public ::testing::Test {
protected:
    TextPreprocessorTest() : preprocessor_("../../weights/tokenizer/vocab.json") {}

    PromptDetector::TextPreprocessor preprocessor_;
};

TEST_F(TextPreprocessorTest, HandlesFourWordPromptCorrectly) {
    std::string prompt = "person car dog cat";
    std::vector<int32_t> token_ids;
    std::vector<int32_t> attention_mask;

    preprocessor_.Process(prompt, token_ids, attention_mask);

    const int start_id = 49406;
    const int end_id = 49407;

    std::vector<int32_t> expected_tokens = {
        start_id,2533, 1615, 1929, 2368, end_id,
        start_id, end_id, end_id, end_id, end_id, end_id
    };

    std::vector<int32_t> expected_mask = {
        1, 1, 1, 1, 1, 1,
        1, 1, 0, 0, 0, 0
    };

    ASSERT_EQ(token_ids, expected_tokens);
    ASSERT_EQ(attention_mask, expected_mask);
}

TEST_F(TextPreprocessorTest, HandlesThreeWordPromptWithPadding) {
    std::string prompt = "person car dog";
    std::vector<int32_t> token_ids;
    std::vector<int32_t> attention_mask;

    preprocessor_.Process(prompt, token_ids, attention_mask);

    const int start_id = 49406;
    const int end_id = 49407;

    std::vector<int32_t> expected_tokens = {
        start_id, 2533, 1615, 1929, end_id, end_id,
        start_id, end_id, end_id, end_id, end_id, end_id
    };

    std::vector<int32_t> expected_mask = {
        1, 1, 1, 1, 1, 0,
        1, 1, 0, 0, 0, 0
    };

    ASSERT_EQ(token_ids, expected_tokens);
    ASSERT_EQ(attention_mask, expected_mask);
}
