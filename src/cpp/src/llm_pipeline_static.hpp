// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>

#include "llm_pipeline_base.hpp"
#include "sampler.hpp"

namespace ov {
namespace genai {

struct ModelConfigDesc {
    std::string type;
    std::string name_or_path;
    int num_key_value_heads;
};

class StaticLLMPipeline final : public LLMPipelineImplBase {
public:
    StaticLLMPipeline(
        const std::filesystem::path& path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& config
    );

    StaticLLMPipeline(
        const std::shared_ptr<ov::Model>& model,
        const ModelConfigDesc& model_desc,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& properties,
        const ov::genai::GenerationConfig& generation_config = {}
    );

    StaticLLMPipeline(
        const std::filesystem::path& path,
        const std::string& device,
        const ov::AnyMap& config
    );

    void setupAndCompileModels(
        const std::shared_ptr<ov::Model>& model,
        const std::string& device,
        const ModelConfigDesc& model_desc,
        ov::AnyMap& pipeline_config);

    void setupAndImportModels(
        const std::filesystem::path& path,
        const std::string& device,
        ov::AnyMap& pipeline_config);

    DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    void start_chat(const std::string& system_message) override;
    void finish_chat() override;
private:
    void prepare_for_new_conversation();

private:
    struct KVCacheDesc {
        uint32_t max_prompt_size;
        uint32_t total_size;
        uint32_t num_stored_tokens;
        uint32_t seq_len;
        bool v_tensors_transposed;
    };

    Sampler m_sampler;

    KVCacheDesc m_kvcache_desc;
    ov::InferRequest m_kvcache_request;
    ov::InferRequest m_prefill_request;

    bool m_is_chat_conversation = false;
    ChatHistory m_history;
};

}  // namespace genai
}  // namespace ov
