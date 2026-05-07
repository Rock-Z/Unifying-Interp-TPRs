"""Utilities for loading paired occupation sentences datasets."""

from datasets import Dataset, DatasetDict
from typing import List, Tuple
import os


class PairedRoleAssigner:
    """Assign roles and filler indices for paired occupation sentences."""

    def __init__(self, occupations: List[str], verbs: List[str]):
        self.occupations = list(dict.fromkeys(occupations))
        self.verbs = list(dict.fromkeys(verbs))
        self.occ_filler2idx = {o: i for i, o in enumerate(self.occupations)}
        self.idx2occ_filler = {i: o for i, o in enumerate(self.occupations)}
        self.role2idx = {v: i for i, v in enumerate(self.verbs)}
        self.idx2role = {i: v for i, v in enumerate(self.verbs)}

    @classmethod
    def from_dataset(cls, path: str, prefix: str) -> "PairedRoleAssigner":
        """Load vocabularies from dataset files."""
        occupations: List[str] = []
        verbs: List[str] = []
        with open(os.path.join(path, f"{prefix}.occupations"), "r") as f:
            for line in f:
                occupations.append(line.strip())
        with open(os.path.join(path, f"{prefix}.verbs"), "r") as f:
            for line in f:
                verbs.append(line.strip())
        return cls(occupations, verbs)

    def get_roles(self, sentence: str) -> dict:
        """Return filler and role ids for the provided sentence.
        
        For paired sentences, roles correspond to the verbs being performed:
        - Role ids map to the actual verbs that appear in each part
        - Filler ids map to the occupations in each part
        """
        sentence = sentence.rstrip(".")
        first, second = sentence.split(". ")
        words1 = first.split()
        words2 = second.split()
        occ1 = words1[1]  # "The [occ1] will verb1"
        verb1 = words1[3]  # "The occ1 will [verb1]"
        occ2 = words2[1]  # "The [occ2] will verb2"
        verb2 = words2[3]  # "The occ2 will [verb2]"
        
        fillers = (
            self.occ_filler2idx[occ1],
            self.occ_filler2idx[occ2],
        )
        roles = (
            self.role2idx[verb1],
            self.role2idx[verb2],
        )
        return {"filler_ids": fillers, "role_ids": roles}


def load_paired_sentences(sentences_path: str, role_scheme: str = "pair") -> Tuple[DatasetDict, PairedRoleAssigner]:
    """Load the paired sentences dataset and attach role annotations.

    The ``role_scheme`` argument is accepted for API compatibility with
    ``load_sentences`` but is currently unused.
    """
    dataset = {}
    role_assigner = PairedRoleAssigner.from_dataset(sentences_path, "data")
    for split in ["train", "valid", "test"]:
        with open(f"{sentences_path}/data.{split}", "r") as f:
            lines = f.readlines()[1:]
        sents = [l.strip() for l in lines]
        ds = Dataset.from_dict({"sentence": sents})
        ds = ds.map(lambda x: role_assigner.get_roles(x["sentence"]))
        dataset[split] = ds
    return DatasetDict(dataset), role_assigner
