import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from webapp.model import VirgoModel

class VirgoInference:
    def __init__(self, model_path, tokenizer_path, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        print("✅ Tokenizer loaded successfully!")
        
        # Load Model Weights
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        
        # Constants from training config
        VOCAB_SIZE = 45000
        D_MODEL = 768
        NUM_HEADS = 12
        NUM_LAYERS = 12
        D_FF = 3072
        MAX_SEQ_LENGTH = 1024
        DROPOUT = 0.0 # eval mode
        
        self.max_seq_length = MAX_SEQ_LENGTH
        
        self.model = VirgoModel(
            vocab_size=VOCAB_SIZE,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            d_ff=D_FF,
            max_seq_length=MAX_SEQ_LENGTH,
            dropout=DROPOUT
        )
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.stop_requested = False
        print("✅ Virgo Model loaded successfully!")

    def _format_chat_prompt(self, user_message, history=None):
        """Wrap user message and history in the Virgo chat template."""
        if history is None:
            history = []
        
        prompt = "<bos>"
        for msg in history:
            role = "User" if msg.get("role") == "user" else "Virgo"
            prompt += f"<newline>{role}<newline>{msg.get('content', '')}<newline><newline>"
            
        prompt += f"<newline>User<newline>{user_message}<newline><newline>Virgo<newline>"
        return prompt

    def _clean_response(self, text):
        """Strip special tokens and extract Virgo's response from raw output."""
        # Split specifically on the prompt turn header marker
        if "<newline>Virgo<newline>" in text:
            text = text.split("<newline>Virgo<newline>")[-1]
        elif "Virgo\n" in text:
            text = text.split("Virgo\n")[-1]

        # Remove <eos> and anything after
        if "<eos>" in text:
            text = text.split("<eos>")[0]

        # Replace special tokens with their real characters
        text = text.replace("<newline>", "\n")
        text = text.replace("<tab>", "\t")
        text = text.replace("<bos>", "")
        text = text.replace("<pad>", "")
        text = text.replace("<unk>", "")
        text = text.replace("<raw_token>", "")

        cleaned = text.strip()
        # Truncate incomplete trailing sentences if response was cut off mid-sentence
        valid_endings = (".", "!", "?", "```", '"', "'", "}", ")", "]", "\n", ":")
        if cleaned and not cleaned.endswith(valid_endings):
            last_punc = max(cleaned.rfind('.'), cleaned.rfind('!'), cleaned.rfind('?'))
            if last_punc > 0:
                cleaned = cleaned[:last_punc + 1]

        return cleaned

    @torch.no_grad()
    def generate(
        self,
        prompt,
        history=None,
        max_new_tokens=128,
        temperature=0.0,
        top_k=1,
        top_p=0.9,
        repetition_penalty=1.1,
    ):
        if history is None:
            history = []
            
        # Iteratively drop history if it exceeds context limit
        while True:
            formatted_prompt = self._format_chat_prompt(prompt, history)
            encoding = self.tokenizer.encode(formatted_prompt)
            ids = encoding.ids
            
            if len(ids) + max_new_tokens <= self.max_seq_length or len(history) == 0:
                break
                
            # Drop the oldest user-system exchange (first 2 items)
            if len(history) >= 2:
                history = history[2:]
            else:
                history = []

        # Prevent empty prompt
        if len(ids) == 0:
            bos_id = self.tokenizer.token_to_id("<bos>")
            if bos_id is None:
                raise ValueError("Prompt produced zero tokens and tokenizer has no <bos> token.")
            ids = [bos_id]

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        generated = input_ids.clone()

        eos_id = self.tokenizer.token_to_id("<eos>")

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if getattr(self, 'stop_requested', False):
                    self.stop_requested = False
                    break
                
                # Sliding context
                context = generated[:, -self.max_seq_length:]

                logits = self.model(context)
                next_token_logits = logits[:, -1, :].float()

                # Repetition penalty
                if repetition_penalty != 1.0:
                    previous_tokens = torch.unique(generated)
                    next_token_logits[:, previous_tokens] = torch.where(
                        next_token_logits[:, previous_tokens] < 0,
                        next_token_logits[:, previous_tokens] * repetition_penalty,
                        next_token_logits[:, previous_tokens] / repetition_penalty,
                    )

                # Greedy decoding
                if temperature == 0:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                else:
                    next_token_logits /= temperature

                    # Top-k
                    if top_k is not None:
                        k = min(top_k, next_token_logits.size(-1))
                        values, _ = torch.topk(next_token_logits, k)
                        next_token_logits[next_token_logits < values[:, [-1]]] = -float("inf")

                    # Top-p
                    if top_p is not None:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        probs = F.softmax(sorted_logits, dim=-1)
                        cumulative = torch.cumsum(probs, dim=-1)

                        remove = cumulative > top_p
                        remove[:, 1:] = remove[:, :-1].clone()
                        remove[:, 0] = False

                        sorted_logits[remove] = -float("inf")
                        next_token_logits = torch.full_like(next_token_logits, -float("inf"))
                        next_token_logits.scatter_(1, sorted_indices, sorted_logits)

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat((generated, next_token), dim=1)

                if eos_id is not None and next_token.item() == eos_id:
                    break

        # Decode full output and clean the response
        raw_output = self.tokenizer.decode(
            generated[0].tolist(),
            skip_special_tokens=False
        )

        tokens_used = generated.shape[1]
        return self._clean_response(raw_output), tokens_used
