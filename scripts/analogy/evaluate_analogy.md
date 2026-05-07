# Analogy Evaluation for Sentence Embeddings

Evaluates whether embedding models can capture systematic relationships between sentences by testing analogies of the form: **A - B + C ≈ D**

## Examples

**Subject analogies:** "doctor sees nurse" - "police sees nurse" + "police sees teacher" ≈ "doctor sees teacher"

**Object analogies:** "doctor sees nurse" - "doctor sees teacher" + "police sees teacher" ≈ "police sees nurse"

## Usage

```bash
# Run with gin config
uv run src/evaluate_analogy.py configs/evaluate_analogy.gin

# Or programmatically
uv run python -c "
from src.evaluate_analogy import main
results = main(
    sentences_path='sentences/',
    embedding_model_name='nomic-ai/modernbert-embed-base',
    role_scheme='svo'
)
print(f'Top-1 accuracy: {results[\"overall_statistics\"][\"top_1_accuracy\"]:.3f}')
"
```

## Dataset Format

Expects sentences like `"the [SUBJECT] will [VERB] the [OBJECT]."` with dataset structure:
```
sentences/
├── data.test, data.train, data.valid
├── data.nouns  
└── data.verbs
```

## Output

Reports Top-k accuracy and mean rank for subject/object analogies. Good models achieve >0.3 Top-1 accuracy and <10 mean rank. 