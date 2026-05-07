from datasets import Dataset, DatasetDict
import os
import random
from typing import Iterable, Optional, Sequence

from vocabulary import PASSIVE_PARTICIPLES

def load_sentences(sentences_path: str, role_scheme: str = "svo"):

    dataset = {}
    sentence_role_assigner = SVORoleAssigner.from_dataset(sentences_path, "data", role_scheme=role_scheme)
    required_splits = ["train", "valid", "test"]
    optional_splits = ["generalization"]
    available_splits = []

    for split in required_splits:
        if not os.path.exists(f"{sentences_path}/data.{split}"):
            raise FileNotFoundError(f"Missing required sentences split: {sentences_path}/data.{split}")
        available_splits.append(split)
    for split in optional_splits:
        if os.path.exists(f"{sentences_path}/data.{split}"):
            available_splits.append(split)

    # load dataset
    for split in available_splits:
        with open(f"{sentences_path}/data.{split}", "r") as f:
            sentences = f.readlines()
            sentences = sentences[1:]  # skip header
        sentences = [s.strip() for s in sentences]
        dataset_split = Dataset.from_dict({"sentence": sentences})
        dataset_split = dataset_split.map(lambda x: sentence_role_assigner.get_roles(x["sentence"]))
        dataset[split] = dataset_split

    # create dataset dict
    dataset = DatasetDict(dataset)

    return dataset, sentence_role_assigner


def is_single_token(tokenizer, word: str) -> bool:
    """Return True when ``\" \" + word`` maps to a single token id."""

    token_ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
    return len(token_ids) == 1


def filter_words_single_token(tokenizer, words: Iterable[str]) -> list[str]:
    """Filter ``words`` down to those that remain single-token under ``tokenizer``."""

    return [word for word in words if is_single_token(tokenizer, word)]


def build_active_passive_trial(A: str, B: str, verb: str) -> dict[str, str]:
    """Construct an IOI-style active+passive prompt."""

    if A == B:
        raise ValueError("Trial requires distinct nouns for A and B")
    if verb not in PASSIVE_PARTICIPLES:
        raise KeyError(f"Verb '{verb}' missing passive participle entry")
    vbn = PASSIVE_PARTICIPLES[verb]
    prompt = f"the {A} will {verb} the {B}. the {B} will be {vbn} by the"
    return {
        "A": A,
        "B": B,
        "verb": verb,
        "vbn": vbn,
        "prompt": prompt,
        "target_correct": " " + A,
        "target_competitor": " " + B,
    }


def sample_trials(
    nouns: Sequence[str], verbs: Sequence[str], n: Optional[int], seed: int
) -> list[dict[str, str]]:
    """Assemble candidate active+passive trials and optionally subsample."""

    rng = random.Random(seed)
    pool: list[dict[str, str]] = []
    noun_list = list(nouns)
    verb_list = list(verbs)
    for A in noun_list:
        for B in noun_list:
            if A == B:
                continue
            for verb in verb_list:
                if verb not in PASSIVE_PARTICIPLES:
                    continue
                pool.append(build_active_passive_trial(A, B, verb))
    if not pool:
        raise ValueError("No valid trials constructed from provided nouns/verbs")
    if n is None or n >= len(pool):
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, n)


def build_active_passive_prompts_from_svo(
    dataset: DatasetDict,
    role_assigner: "SVORoleAssigner",
    tokenizer,
    max_examples_per_split: Optional[int] = None,
) -> DatasetDict:
    """Convert SVO sentences into concatenated active+passive prompts.

    Args:
        dataset: DatasetDict with splits containing ``sentence``, ``filler_ids``,
            and ``role_ids`` for SVO-format sentences.
        role_assigner: SVORoleAssigner providing noun/verb lookups and role ids.
        tokenizer: HF tokenizer used to enforce single-token nouns.
        max_examples_per_split: Optional cap on examples per split after filtering.

    Returns:
        A DatasetDict with the same splits, each row containing:
        - ``sentence``: "the SUBJ will VERB the OBJ. the OBJ will be VBN by the SUBJ"
        - ``filler_ids`` and ``role_ids`` copied from the original dataset.

    Example:
        Input row: ``{"sentence": "the chef will see the nurse.", "filler_ids": (1, 2, 5), "role_ids": (0, 2, 1)}``
        Output row: ``{"sentence": "the chef will see the nurse. the nurse will be seen by the chef",
                       "filler_ids": (1, 2, 5), "role_ids": (0, 2, 1)}``
    """

    noun_vocab = len(role_assigner.noun_idx2filler)
    output = {}
    for split_name, split_ds in dataset.items():
        records = []
        for ex in split_ds:
            subj_id, obj_id, verb_id = ex["filler_ids"]
            verb_idx = verb_id - noun_vocab
            subj = role_assigner.noun_idx2filler[subj_id]
            obj = role_assigner.noun_idx2filler[obj_id]
            verb = role_assigner.verb_idx2filler[verb_idx]
            if subj == obj:
                continue
            if verb not in PASSIVE_PARTICIPLES:
                continue
            if not is_single_token(tokenizer, subj) or not is_single_token(tokenizer, obj):
                continue
            vbn = PASSIVE_PARTICIPLES[verb]
            prompt = f"the {subj} will {verb} the {obj}. the {obj} will be {vbn} by the {subj}"
            records.append(
                {
                    "sentence": prompt,
                    "filler_ids": ex["filler_ids"],
                    "role_ids": ex["role_ids"],
                }
            )
            if max_examples_per_split is not None and len(records) >= max_examples_per_split:
                break
        output[split_name] = Dataset.from_list(records)
    return DatasetDict(output)

class SVORoleAssigner:
    """A class for assigning and mapping semantic roles in subject-verb-object sentences.
    
    This class handles the mapping between words (fillers) and their roles in sentences
    that follow the pattern "the [subject] will [verb] the [object]."
    
    Attributes:
        nouns_sg: List of singular nouns that can be subjects or objects
        verbs: List of verbs that can appear in sentences
        noun_filler2idx: Mapping from noun words to their indices
        noun_idx2filler: Mapping from indices to noun words
        verb_filler2idx: Mapping from verb words to their indices
        verb_idx2filler: Mapping from indices to verb words
        role2idx: Mapping from role names to their indices
        idx2role: Mapping from indices to role names
    """
    
    def __init__(self, nouns_sg: list[str], verbs: list[str], role_scheme: str = "svo"):
        """Initialize the role assigner with vocabularies of nouns and verbs.
        
        Args:
            nouns_sg: List of singular nouns that can be subjects or objects
            verbs: List of verbs that can appear in sentences
        """
        self.nouns_sg = list(set(nouns_sg))  # remove duplicates
        self.verbs = list(set(verbs)) 
        
        self.noun_filler2idx = {noun: i for i, noun in enumerate(nouns_sg)}
        self.noun_idx2filler = {i: noun for i, noun in enumerate(nouns_sg)}
        self.verb_filler2idx = {verb: i for i, verb in enumerate(verbs)}
        self.verb_idx2filler = {i: verb for i, verb in enumerate(verbs)}

        self.role2idx = {"subject": 0, "verb": 1, "object": 2} if role_scheme == "svo" else {"bow": 0}
        self.idx2role = {v: k for k, v in self.role2idx.items()}
        self.role_scheme = role_scheme
        # Reverse mapping for passive participles -> base verb
        self.passive_participle_to_base = {v: k for k, v in PASSIVE_PARTICIPLES.items()}

    @classmethod
    def from_dataset(cls, path: str, prefix: str, role_scheme: str = "svo") -> 'SVORoleAssigner':
        """Loads role assigner and subject/verb vocabularies from provided path.

        Args:
            path: Directory path where vocabulary files are stored
            prefix: File prefix for the vocabulary files

        Returns:
            SVORoleAssigner: An instance initialized with vocabularies from files
            
        Raises:
            FileNotFoundError: If vocabulary files do not exist at specified location
        """
        # Load nouns and verbs
        nouns_sg = []
        verbs = []
        with open(os.path.join(path, f"{prefix}.nouns"), "r") as f:
            for line in f:
                nouns_sg.append(line.strip())
        with open(os.path.join(path, f"{prefix}.verbs"), "r") as f:
            for line in f:
                verbs.append(line.strip())
        
        return cls(nouns_sg, verbs, role_scheme=role_scheme)
    
    def get_roles(self, sentence: str) -> dict[str, tuple[int, int, int]]:
        """Returns filler and role indices for subject, verb, object.

        Supports both active voice:
            "the SUBJ will VERB the OBJ ."
        and passive voice:
            "the OBJ will be VERB-PART by the SUBJ ."
        """
        words = sentence.split()
        # Normalize trailing punctuation: strip a single final period from the last token
        if words and words[-1].endswith('.'):
            words[-1] = words[-1][:-1]
        # Heuristic: passive if token "be" appears after "will" and token "by" exists
        passive = False
        if len(words) >= 8:
            try:
                will_idx = words.index("will")
                passive = (will_idx + 1 < len(words) and words[will_idx + 1] == "be" and "by" in words)
            except ValueError:
                passive = False
        if passive:
            # the OBJ will be VBN by the SUBJ .
            obj = words[1]
            # words[will_idx+2] is the participle; we need to map it back to base verb form if possible
            v_part = words[words.index("will") + 2]
            # Reverse-map participle to base verb
            if v_part in self.passive_participle_to_base:
                verb = self.passive_participle_to_base[v_part]
            else:
                raise ValueError(f"Unknown participle in passive sentence: {v_part}")
            # Skip determiner after 'by'
            by_idx = words.index("by")
            subj = words[by_idx + 2]
        else:
            # Active pattern: the SUBJ will VERB the OBJ .
            subj = words[1]
            obj = words[5]
            verb = words[3]
        if subj not in self.noun_filler2idx or obj not in self.noun_filler2idx or verb not in self.verb_filler2idx:
            raise ValueError(f"Unknown words in {sentence=}: {subj=}, {obj=}, {verb=}")
        subj_filler = self.noun_filler2idx[subj]
        obj_filler = self.noun_filler2idx[obj]
        verb_filler = self.verb_filler2idx[verb] + len(self.nouns_sg)
        if self.role_scheme == "bow":
            role_ids = (0, 0, 0)
        else:  # default to SVO
            role_ids = (self.role2idx["subject"], self.role2idx["object"], self.role2idx["verb"])
        return {
            "filler_ids": (subj_filler, obj_filler, verb_filler),
            "role_ids": role_ids
        }
    
    def reconstruct_sentence(self, fillers: tuple[int, int, int], roles: tuple[int, int, int]) -> str:
        """Reconstructs a sentence from filler and role indices.
        
        Takes the indices of fillers and their corresponding roles, and reconstructs
        a sentence following the pattern "the [subject] will [verb] the [object]."
        
        Args:
            fillers: A tuple of indices representing the words to fill in roles
            roles: A tuple of indices representing the roles (subject, verb, object)
            
        Returns:
            str: A reconstructed sentence in the format "the [subject] will [verb] the [object]."
            
        Raises:
            KeyError: If the provided indices don't exist in the vocabulary mappings
            IndexError: If the provided tuples are not the expected length
        """
        svo = {
            self.idx2role[roles[i]]: self.noun_idx2filler[fillers[i]] 
            if roles[i] != self.role2idx["verb"] else self.verb_idx2filler[fillers[i] - len(self.nouns_sg)]
            for i in range(len(roles))
        }

        return f"the {svo['subject']} will {svo['verb']} the {svo['object']}."
