import pytest
import torch

from minrl.model import LanguageModel, generate
from minrl.tokenizer import Tokenizer
from transformers import Qwen2ForCausalLM


DEVICE = "cuda:0"
DTYPE = torch.float16
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="session")
def reference_model():
    return Qwen2ForCausalLM.from_pretrained(MODEL_NAME).to(device=DEVICE, dtype=DTYPE).eval()


@pytest.fixture(scope="session")
def our_model():
    return (
        LanguageModel.from_hf(MODEL_NAME, device=DEVICE, dtype=DTYPE).eval().init_kv_cache(1, 128)
    )


@pytest.fixture(scope="session")
def tokenizer():
    return Tokenizer.from_huggingface(MODEL_NAME)


@torch.no_grad
def test_qwen2_outputs_match_reference(
    reference_model: Qwen2ForCausalLM,
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

    torch.testing.assert_close(ref_tensors, our_tensors, rtol=0, atol=0.01)
    torch.testing.assert_close(ref_output, our_output, rtol=0, atol=0.01)

    for handle in handle_list:
        handle.remove()


@torch.no_grad
def test_language_model_decoding_match_training(our_model: LanguageModel, tokenizer: Tokenizer):
    msg = "Solve for x: x*13 + 46 = 345"
    chat = tokenizer.encode_chat([{"role": "user", "content": msg}])
    encoding = tokenizer.tokenize(chat)

    intermediate_activations = {}

    # Register forward hooks
    def hook(name):
        def inner_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if name not in intermediate_activations:
                intermediate_activations[name] = []
            intermediate_activations[name].append(output)

        return inner_hook

    hook_list = [
        (our_model.layers[0].input_layernorm, "layernorm_0"),
        (our_model.layers[0].self_attn, "attn_0"),
        (our_model.layers[0].mlp, "mlp_0"),
        (our_model.layers[1].input_layernorm, "layernorm_1"),
        (our_model.layers[1].self_attn, "attn_1"),
        # (our_model.layers[1].mlp, "mlp_1"),
    ]

    handle_list = []
    for module, name in hook_list:
        handle_list += [module.register_forward_hook(hook(name))]

    input_ids = torch.tensor(encoding.ids, device=DEVICE, dtype=torch.int64)[None]
    y_training = our_model.forward(input_ids)

    y_decoding_list = []
    cache_pos = torch.tensor([0], device=DEVICE, dtype=torch.int32)
    pos = 0
    next_pos = 16
    while next_pos <= input_ids.shape[1]:
        y = our_model.forward(input_ids[:, pos:next_pos], cache_pos=cache_pos)
        y_decoding_list.append(y)
        pos = next_pos
        cache_pos[0] = pos
        next_pos += 1

    x_training = {k: intermediate_activations[k][0] for k in intermediate_activations}
    x_decoding = {
        k: torch.cat(intermediate_activations[k][1:], dim=1) for k in intermediate_activations
    }
    torch.testing.assert_close(x_decoding, x_training, rtol=0, atol=0.01)

    y_decoding = torch.cat(y_decoding_list, dim=1)
    mse = torch.nn.functional.mse_loss(y_decoding, y_training).item()
    assert mse < 0.001, f"Decoding and training outputs do not match: {mse=}"

    for handle in handle_list:
        handle.remove()


@torch.no_grad
def test_language_model_can_generate(our_model: LanguageModel, tokenizer: Tokenizer):
    msg = "Solve for x: x*13 + 46 = 345"
    chat = tokenizer.encode_chat([{"role": "user", "content": msg}])
    encoding = tokenizer.tokenize(chat)
    input_ids = torch.tensor(encoding.ids, device=DEVICE, dtype=torch.int64)[None]
    generate(our_model, input_ids, generation_length=10)


@torch.no_grad
def test_language_model_generate_raises_with_invalid_padding(our_model: LanguageModel):
    # Padding in between
    input_ids = torch.tensor([0, 2, 0, 4], device=DEVICE, dtype=torch.int64)[None]
    with pytest.raises(ValueError):
        generate(our_model, input_ids, generation_length=10, pad_token_id=0)

    # Padding at the beginning
    input_ids = torch.tensor([0, 0, 2, 4], device=DEVICE, dtype=torch.int64)[None]
    with pytest.raises(ValueError):
        generate(our_model, input_ids, generation_length=10, pad_token_id=0)

    # Padding at the end
    input_ids = torch.tensor([2, 4, 0, 0], device=DEVICE, dtype=torch.int64)[None]
    generate(our_model, input_ids, generation_length=10, pad_token_id=0)
