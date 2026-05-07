import argparse
import json
import os
import random

from vocabulary import NOUNS_SG, VERBS, MULTIPLE_VERBS, PASSIVE_PARTICIPLES

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", help="prefix of file to save the sequences to", type=str, default="sentences/data")
    parser.add_argument("--verb_set", help="verb set to use", type=str, choices=["single_verb", "multiple_verbs"], default="single_verb")
    parser.add_argument("--voice", help="sentence voice to generate", type=str, choices=["active", "passive"], default="active")
    parser.add_argument("--seed", help="optional random seed used for split shuffling", type=int, default=None)
    parser.add_argument("--holdout_step", help="stride for selecting held-out nouns; disabled when <= 0", type=int, default=0)
    parser.add_argument("--subject_holdout_offset", help="offset into the noun list for subject-role holdouts", type=int, default=None)
    parser.add_argument("--object_holdout_offset", help="offset into the noun list for object-role holdouts", type=int, default=None)
    parser.add_argument("--verbose", help="print generation statistics", action="store_true", default=True)
    return parser.parse_args()

def generate_sentences(nouns_sg, verbs, voice: str = "active"):
    """Generate sentences with the provided nouns and verbs.

    voice: "active" -> "the subj will verb the obj ."
           "passive" -> "the obj will be VBN by the subj ."
    """
    sentences = []
    for subj in nouns_sg:
        for dobj in nouns_sg:
            for verb in verbs:
                if voice == "active":
                    base = " ".join(["the", subj, "will", verb, "the", dobj])
                    sentence = base + "."
                else:
                    # Passive voice requires a participle; skip verbs without a mapping
                    if verb not in PASSIVE_PARTICIPLES:
                        continue
                    vbn = PASSIVE_PARTICIPLES[verb]
                    base = " ".join(["the", dobj, "will", "be", vbn, "by", "the", subj])
                    sentence = base + "."
                sentences.append(sentence)
    return sentences

def split_data(sentences, seed: int | None = None):
    sentences = list(sentences)
    if seed is None:
        random.shuffle(sentences)
    else:
        random.Random(seed).shuffle(sentences)
    count = len(sentences)
    train_set = sentences[:int(count * 0.8)]
    valid_set = sentences[int(count * 0.8):int(count * 0.9)]
    test_set = sentences[int(count*0.9):]
    return train_set, valid_set, test_set

def save_sentences(prefix, train_set, valid_set, test_set):
    # Ensure directory exists
    dirpath = os.path.dirname(prefix)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    with open(prefix + ".train", "w") as fo_train:
        fo_train.write("sentence\n")
        for line in train_set:
            fo_train.write(line + "\n")
    
    with open(prefix + ".valid", "w") as fo_valid:
        fo_valid.write("sentence\n")
        for line in valid_set:
            fo_valid.write(line + "\n")
    
    with open(prefix + ".test", "w") as fo_test:
        fo_test.write("sentence\n")
        for line in test_set:
            fo_test.write(line + "\n")

def save_split(prefix: str, split_name: str, sentences: list[str]) -> None:
    dirpath = os.path.dirname(prefix)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    with open(prefix + f".{split_name}", "w") as fo:
        fo.write("sentence\n")
        for line in sentences:
            fo.write(line + "\n")

def save_word_lists(prefix, nouns_sg, verbs):
    # Ensure directory exists
    dirpath = os.path.dirname(prefix)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    with open(prefix + ".nouns", "w") as fo:
        for noun in nouns_sg:
            fo.write(noun + "\n")
    with open(prefix + ".verbs", "w") as fo:
        for verb in verbs:
            fo.write(verb + "\n")

def get_holdout_nouns(nouns_sg: list[str], step: int, offset: int) -> list[str]:
    """Select held-out nouns by taking every ``step`` entries starting at ``offset``."""

    if step <= 0:
        raise ValueError("holdout step must be positive")
    if offset < 0 or offset >= step:
        raise ValueError(f"holdout offset must satisfy 0 <= offset < step; got {offset=} and {step=}")
    return list(nouns_sg[offset::step])

def extract_subject_and_object(sentence: str, voice: str) -> tuple[str, str]:
    """Return the semantic subject and object from a generated sentence string."""

    words = sentence.rstrip(".").split()
    if voice == "active":
        return words[1], words[5]
    if voice == "passive":
        return words[7], words[1]
    raise ValueError(f"Unsupported voice: {voice}")

def split_data_with_holdout(
    sentences: list[str],
    subject_holdout_nouns: list[str],
    object_holdout_nouns: list[str],
    *,
    voice: str,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Partition sentences into in-distribution splits plus a generalization split."""

    subject_holdouts = set(subject_holdout_nouns)
    object_holdouts = set(object_holdout_nouns)
    in_distribution = []
    generalization = []

    for sentence in sentences:
        subject, obj = extract_subject_and_object(sentence, voice=voice)
        if subject in subject_holdouts or obj in object_holdouts:
            generalization.append(sentence)
        else:
            in_distribution.append(sentence)

    train_set, valid_set, test_set = split_data(in_distribution, seed=seed)
    return {
        "train": train_set,
        "valid": valid_set,
        "test": test_set,
        "generalization": generalization,
    }

def build_holdout_metadata(
    splits: dict[str, list[str]],
    *,
    subject_holdout_nouns: list[str],
    object_holdout_nouns: list[str],
    holdout_step: int,
    subject_holdout_offset: int,
    object_holdout_offset: int,
    voice: str,
) -> dict:
    """Summarize how a sentence filler-role holdout dataset was constructed."""

    return {
        "holdout_step": holdout_step,
        "subject_holdout_offset": subject_holdout_offset,
        "object_holdout_offset": object_holdout_offset,
        "voice": voice,
        "subject_holdout_nouns": list(subject_holdout_nouns),
        "object_holdout_nouns": list(object_holdout_nouns),
        "split_sizes": {split_name: len(split_sentences) for split_name, split_sentences in splits.items()},
        "examples_with_holdout_pair": {
            split_name: (
                len(split_sentences)
                if split_name == "generalization"
                else 0
            )
            for split_name, split_sentences in splits.items()
        },
    }

def save_holdout_metadata(prefix: str, metadata: dict) -> None:
    with open(prefix + ".holdout_metadata.json", "w") as fo:
        json.dump(metadata, fo, indent=2)

def main():
    args = parse_args()
    
    # Select the appropriate verb list based on verb_set
    if args.verb_set == "single_verb":
        verbs = VERBS
    elif args.verb_set == "multiple_verbs":
        verbs = MULTIPLE_VERBS
    else:
        raise ValueError(f"Unknown verb_set: {args.verb_set}. Must be 'single_verb' or 'multiple_verbs'")
    
    # Generate all possible sentences
    sentences = generate_sentences(NOUNS_SG, verbs, voice=args.voice)
    
    if args.verbose:
        print(f"Generated {len(sentences)} sentences using {args.verb_set} verb set in {args.voice} voice")
        print(f"Verbs used ({len(verbs)}): {', '.join(verbs)}")

    use_holdout_split = args.holdout_step > 0
    if use_holdout_split:
        if args.subject_holdout_offset is None or args.object_holdout_offset is None:
            raise ValueError(
                "Holdout generation requires --subject_holdout_offset and --object_holdout_offset."
            )
        subject_holdout_nouns = get_holdout_nouns(
            NOUNS_SG, args.holdout_step, args.subject_holdout_offset
        )
        object_holdout_nouns = get_holdout_nouns(
            NOUNS_SG, args.holdout_step, args.object_holdout_offset
        )
        splits = split_data_with_holdout(
            sentences,
            subject_holdout_nouns,
            object_holdout_nouns,
            voice=args.voice,
            seed=args.seed,
        )
        for split_name, split_sentences in splits.items():
            save_split(args.prefix, split_name, split_sentences)
        metadata = build_holdout_metadata(
            splits,
            subject_holdout_nouns=subject_holdout_nouns,
            object_holdout_nouns=object_holdout_nouns,
            holdout_step=args.holdout_step,
            subject_holdout_offset=args.subject_holdout_offset,
            object_holdout_offset=args.object_holdout_offset,
            voice=args.voice,
        )
        save_holdout_metadata(args.prefix, metadata)
        if args.verbose:
            print(
                "Holdout nouns:"
                f" subject={subject_holdout_nouns}, object={object_holdout_nouns}"
            )
            print(
                "Split sizes:"
                f" train={len(splits['train'])}, valid={len(splits['valid'])},"
                f" test={len(splits['test'])}, generalization={len(splits['generalization'])}"
            )
    else:
        # Split into train, validation and test sets
        train_set, valid_set, test_set = split_data(sentences, seed=args.seed)

        # Save sentences to files
        save_sentences(args.prefix, train_set, valid_set, test_set)
    
    # Save word lists
    save_word_lists(args.prefix, NOUNS_SG, verbs)

if __name__ == "__main__":
    main()
