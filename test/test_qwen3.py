import pytest
import torch

from minrl.model import LanguageModel
from minrl.tokenizer import Tokenizer
from transformers import Qwen3ForCausalLM


DEVICE = "cuda:0"
DTYPE = torch.float16
MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def reference_model():
    return Qwen3ForCausalLM.from_pretrained(MODEL_NAME).to(device=DEVICE, dtype=DTYPE).eval()


@pytest.fixture(scope="session")
def our_model():
    return (
        LanguageModel.from_hf(MODEL_NAME, device=DEVICE, dtype=DTYPE).eval().init_kv_cache(1, 128)
    )


@pytest.fixture(scope="session")
def tokenizer():
    return Tokenizer.from_huggingface(MODEL_NAME)


@torch.no_grad
def test_qwen3_outputs_match_reference(
    reference_model: Qwen3ForCausalLM,
    our_model: LanguageModel,
    tokenizer: Tokenizer,
):
    msg = "Solve for x: x*13 + 46 = 345"
    chat = tokenizer.encode_chat([{"role": "user", "content": msg}])
    encoding = tokenizer.tokenize(chat)
    input_ids = torch.tensor(encoding.ids, device=DEVICE, dtype=torch.int64)[None]

    our_tensors = {}
    ref_tensors = {}

    # Register forward hooks
    def hook(result, name):
        def inner_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            result[name] = output

        return inner_hook

    hook_list = [
        (reference_model.model.layers[0].input_layernorm, ref_tensors, "layernorm_0"),
        (reference_model.model.layers[0].self_attn, ref_tensors, "attn_0"),
        (reference_model.model.layers[0].mlp, ref_tensors, "mlp_0"),
        (our_model.layers[0].input_layernorm, our_tensors, "layernorm_0"),
        (our_model.layers[0].self_attn, our_tensors, "attn_0"),
        (our_model.layers[0].mlp, our_tensors, "mlp_0"),
    ]

    handle_list = []
    for module, result, name in hook_list:
        handle_list += [module.register_forward_hook(hook(result, name))]

    ref_output = reference_model.model(input_ids).last_hidden_state
    our_output = our_model(input_ids)

    torch.testing.assert_close(ref_tensors, our_tensors, rtol=0, atol=1e-4)
    torch.testing.assert_close(ref_output, our_output, rtol=0, atol=1e-4)

    for handle in handle_list:
        handle.remove()
