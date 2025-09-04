import logging
import re
from collections import defaultdict
from itertools import product

from inflect import engine as inflect_engine
from jiwer import (
    AbstractTransform,
    Compose,
    ReduceToListOfListOfWords,
    ReduceToSingleSentence,
    RemoveKaldiNonWords,
    RemoveMultipleSpaces,
    RemoveWhiteSpace,
    Strip,
    ToUpperCase,
    process_words,
)

p = inflect_engine()
logger = logging.getLogger(__name__)


class NormalizeNumbers(AbstractTransform):
    def process_string(self, s: str):
        def replace_decimal(match):
            num_str = match.group(0)
            whole, decimal = num_str.split(".")
            whole_spoken = p.number_to_words(whole, andword="").replace("-", " ")
            decimal_spoken = " ".join(p.number_to_words(d) for d in decimal)
            return f"{whole_spoken} point {decimal_spoken}"

        def replace_integer(match):
            num_str = match.group(0)
            return p.number_to_words(num_str, andword="").replace("-", " ")

        # Replace decimals first (to avoid partial integer matches)
        text = re.sub(r"\d+\.\d+", replace_decimal, s)

        # Then replace remaining integers
        text = re.sub(r"\b\d+\b", replace_integer, text)
        return text.lower()


class MyRemovePunctuation(AbstractTransform):
    """Replacement for jiwer's remove punctuation that allows more control."""

    def __init__(self, symbols):
        self.substitutions = f"[{symbols}]"

    def process_string(self, s):
        return re.sub(self.substitutions, "", s)


class Contractions(AbstractTransform):
    """Class to handle contractions, e.g. don't vs do not"""

    def __init__(self, alternative_file: str):
        self.alternative_dict = defaultdict(list)
        with open(alternative_file) as f:
            for line in f:
                parts = [x.strip() for x in line.strip().split(",", 1)]
                if len(parts) == 1:
                    k, v = parts[0], ""
                else:
                    k, v = parts
                self.alternative_dict[k.lower()].append(v.lower())

        # Create regex pattern
        pattern = "|".join(
            rf"\b{re.escape(k)}\b" if "'" not in k else rf"(?<!\w){re.escape(k)}(?!\w)"
            for k in self.alternative_dict.keys()
        )

        self.contra_re = re.compile(f"({pattern})", re.IGNORECASE)

    def process_string(self, s):
        """Generate all possible forms of a sentence by expanding using alternatives."""

        APOST = r"['\u2019]"
        token_re = re.compile(
            rf"[a-z]+(?:{APOST}[a-z]+)*(?:{APOST})?|{APOST}[a-z]+|[^\w\s]",
            re.IGNORECASE,
        )

        parts = token_re.findall(s.lower())

        # For each part, list all possible variants
        options = [
            self.alternative_dict[p][0] if p in self.alternative_dict else p
            for p in parts
        ]

        sentence_form = " ".join(options)
        return sentence_form.strip()


class SentenceScorer:
    def __init__(self, contractions_file=None):
        self.transformation = Compose(
            [
                MyRemovePunctuation(";!*#,.′’‘_()"),
                Contractions(contractions_file),
                NormalizeNumbers(),
                RemoveKaldiNonWords(),
                Strip(),
                ToUpperCase(),
                RemoveMultipleSpaces(),
                RemoveWhiteSpace(replace_by_space=True),
                ReduceToSingleSentence(),
            ]
        )

        self.transformation_towords = Compose(
            [ReduceToListOfListOfWords(word_delimiter=" ")]
        )

    def get_word_sequence(self, sentence):
        return self.transformation(sentence)

    def score(self, ref, hyp):
        if isinstance(ref, str):
            _ref_form = [ref]
        else:
            _ref_form = ref.copy()

        if isinstance(hyp, str):
            _hyp_forms = [hyp]
        else:
            _hyp_forms = hyp.copy()

        hyp_forms, ref_forms = [], []

        for hyp_form in _hyp_forms:
            hyp_forms.append(self.transformation(hyp_form))

        for ref_form in _ref_form:
            ref_forms.append(self.transformation(ref_form))

        alternatives = [(x, y) for x, y in product(hyp_forms, ref_forms)]

        measures = [
            process_words(
                ref,
                hyp,
                reference_transform=self.transformation_towords,
                hypothesis_transform=self.transformation_towords,
            )
            for hyp, ref in alternatives
        ]
        hits = [m.hits for m in measures]
        best_index = hits.index(max(hits))

        return measures[best_index]
