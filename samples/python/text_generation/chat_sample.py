#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    args = parser.parse_args()

    device = 'NPU'  # GPU can be used as well
    pipe = openvino_genai.LLMPipeline(args.model_dir, device,
    { "NPU_BYPASS_UMD_CACHING" : "YES", "STATIC_PIPELINE" : "STATEFUL", "GENERATE_HINT" : "BEST_PERF",
      "NPUW_DEVICES" : "NPU", "NPUW_DUMP_FULL" : "YES", "NPUW_DUMP_SUBS" : "YES", "NPUW_DUMP_IO" : "YES"})

    

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    pipe.start_chat()
    while True:
        try:
            prompt = input('question:\n')
        except EOFError:
            break
        pipe.generate(prompt, config, streamer)
        print('\n----------')
    pipe.finish_chat()


if '__main__' == __name__:
    main()
