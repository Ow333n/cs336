import regex as re

def train_bpe(input_path, vocab_size, special_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, 'r', encoding = 'utf-8') as f:
        text = f.read()
    # 1. pretokenization
    pretoken_count = {}
    if special_tokens:
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(special_pattern, text)
    else:
        parts = [text]
    for part in parts:
        for match in re.finditer(PAT, part):
            token = match.group()
            encoded = token.encode("utf-8")
            token_bytes = tuple(bytes([i]) for i in encoded)
            pretoken_count[token_bytes] = pretoken_count.get(token_bytes, 0) + 1

    # 2. initialize vocab
    vocab = {}
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size is too small")

    for i in range(256):
        vocab[i] = bytes([i])
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
        
    # 3. save merge information
    merges = []

    # 4. BPE merge
    while len(vocab) < vocab_size:
        # 4.1 每一轮重新统计 pair
        pair_count = {}
        for token_bytes, count in pretoken_count.items():
            for i in range(len(token_bytes)-1):
                pair = (token_bytes[i], token_bytes[i+1])
                pair_count[pair] = pair_count.get(pair, 0) + count
        if not pair_count:
            break
        # 4.2 找这一轮最频繁的 pair
        max_pair = max(pair_count, key=lambda pair:(pair_count[pair], pair))

        # 4.3 记录这次 merge
        merges.append(max_pair)

        # 4.4 新 token 加进 vocab
        vocab[len(vocab)] = b''.join(max_pair)

        # 4.5 对所有 pre-token 应用这个 merge
        new_token_bytes_count = {}
        for token_bytes, count in pretoken_count.items():
            i = 0
            new_token_bytes = []
            while i < len(token_bytes):
                if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i+1]) == max_pair:
                    new_token_bytes.append(b''.join((token_bytes[i], token_bytes[i+1])))
                    i += 2
                else:
                    new_token_bytes.append(token_bytes[i])
                    i += 1
            new_token_bytes = tuple(new_token_bytes)
            new_token_bytes_count[new_token_bytes] = new_token_bytes_count.get(new_token_bytes, 0) + count

        pretoken_count = new_token_bytes_count
    return vocab, merges

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.merges = merges
        self.merge_ranks = {merge: rank for rank, merge in enumerate(self.merges)}
        self.vocab = vocab
        self.special_tokens = special_tokens or []
        self.bytes_to_id = {token_bytes: id for id, token_bytes in vocab.items()}
    def encode(self, text):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretoken = []
        ids = []
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens,key=len,reverse=True)
            special_pattern = "|".join(re.escape(token) for token in sorted_special_tokens)
            parts = re.split(f'({special_pattern})', text)
        else:
            parts = [text]
        for part in parts:
            if self.special_tokens and part in self.special_tokens:
                pretoken.append(part)
            else:
                for match in re.finditer(PAT, part):
                    pretoken.append(match.group())

        for token in pretoken:
            if token in self.special_tokens:
                token_bytes = token.encode('utf-8')
                ids.append(self.bytes_to_id[token_bytes])
                continue
            encode = token.encode('utf-8')
            token_bytes = tuple(bytes([i]) for i in encode)


            while True:
                best_pair = None
                best_rank = float('inf')

                for i in range(len(token_bytes)-1):
                    pair = (token_bytes[i], token_bytes[i+1])
                    if pair in self.merge_ranks:
                        rank = self.merge_ranks[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                if best_pair is None:
                    break
                i = 0
                new_token_bytes = []
                while i<len(token_bytes):
                    if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i+1]) == best_pair:
                        new_token_bytes.append(b''.join((token_bytes[i], token_bytes[i+1])))
                        i += 2
                    else:
                        new_token_bytes.append(token_bytes[i])
                        i += 1
                token_bytes = tuple(new_token_bytes)
            for token in token_bytes:
                ids.append(self.bytes_to_id[token])

        return ids

    def decode(self, ids):
        bytes_string = b''.join(self.vocab[i] for i in ids)
        return bytes_string.decode('utf-8', errors = 'replace')
    def encode_iterable(self, iterable):
        for text in iterable:
            ids = self.encode(text)
            for token_id in ids:
                yield token_id