import numpy as np
from scipy.special import log_softmax


class CTCBeamSearchDecode:
    """CTC Beam Search Decode - trả về k-best kết quả thay thế CTCLabelDecode."""

    def __init__(self, character_dict_path, use_space_char=True, beam_width=10, k_best=5):
        self.beam_width = beam_width
        self.k_best = k_best
        self.character = []
        with open(character_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.character.append(line.strip('\n').strip('\r\n'))
        if use_space_char:
            self.character.append(' ')
        self.blank_idx = len(self.character)

    def beam_search(self, log_probs):
        T, V = log_probs.shape
        beams = [(tuple(), self.blank_idx, 0.0)]

        for t in range(T):
            new_beams = {}
            for prefix, last_char, score in beams:
                top_indices = np.argsort(log_probs[t])[-self.beam_width:]
                for idx in top_indices:
                    new_score = score + log_probs[t][idx]
                    if idx == self.blank_idx:
                        key = (prefix, self.blank_idx)
                    elif idx == last_char:
                        key = (prefix, idx)
                    else:
                        key = (prefix + (idx,), idx)

                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score

            sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:self.beam_width]
            beams = [(p, lc, s) for (p, lc), s in sorted_beams]

        merged = {}
        for prefix, _, score in beams:
            if prefix not in merged or merged[prefix] < score:
                merged[prefix] = score

        results = []
        for prefix, score in sorted(merged.items(), key=lambda x: x[1], reverse=True)[:self.k_best]:
            text = ''.join([self.character[idx] for idx in prefix if idx < len(self.character)])
            results.append((text, float(np.exp(score))))
        return results

    def __call__(self, preds):
        if isinstance(preds, dict):
            preds = preds.get('ctc', preds)
        if hasattr(preds, 'numpy'):
            preds = preds.numpy()
        log_probs_batch = log_softmax(preds, axis=-1)
        all_results = []
        for i in range(log_probs_batch.shape[0]):
            all_results.append(self.beam_search(log_probs_batch[i]))
        return all_results
