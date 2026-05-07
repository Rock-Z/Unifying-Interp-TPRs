"""Shared vocabularies for sentence generation datasets."""

# List of occupations used in both sentence schemas
NOUNS_SG = [
    "professor", "student", "president", "judge", "senator", "secretary", "doctor",
    "lawyer", "scientist", "banker", "tourist", "artist", "author", "actor",
    "athlete", "teacher", "engineer", "accountant", "architect", "chef",
    "photographer", "farmer", "ambassador", "astronaut", "astronomer",
    "blacksmith", "baker", "barber", "biologist", "butler", "chemist",
    "composer", "cartoonist", "coach", "captain", "carpenter", "dancer",
    "director", "drummer", "detective", "explorer", "economist", "editor",
    "governor", "gardener", "illustrator", "intern", "inventor", "journalist",
    "linguist", "manager", "magician", "mayor", "miner", "mathematician",
    "musician", "novelist", "nurse", "painter", "philosopher", "physicist",
    "politician", "programmer", "pilot", "poet", "reporter", "referee",
    "sailor", "spy", "translator", "treasurer", "technician", "tutor",
    "umpire", "violinist", "writer", "librarian"
]

# Single verb used for the basic sentence dataset
VERBS = ["see"]

# Multiple verbs for sentence dataset  
MULTIPLE_VERBS = [
    "see", "help", "visit", "teach", "call", 
    #"hire", "follow", "invite",  "trust", "support", 
    #"train", "guide", "advise", "protect", "inspire"
]

# Things that occupations can do in the paired sentence dataset
THINGS = [
    "repairs", "research", "cooking", "teaching", "analysis",
    "maintenance", "painting", "translation", "design", "surgery",
]

PAIR_VERBS = ["arrive", "help"]

# Passive participle mapping for supported verbs
# Keys are base verb forms; values are past participles used in passive voice
PASSIVE_PARTICIPLES = {
    "see": "seen",
    "help": "helped",
    "visit": "visited",
    "teach": "taught",
    "call": "called",
}
