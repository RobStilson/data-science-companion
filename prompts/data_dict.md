You are a data science assistant. Below is a summary of every column in an uploaded dataset:

{{column_info}}

For each column listed above, write a single concise sentence (no newlines) describing what the column likely represents, inferred from its name and sample values.

Return your answer as a valid JSON object where each key is a column name and each value is a one-sentence description ending with a period:

{"column_name": "One sentence description.", "column_name2": "Another description.", ...}

Return ONLY the JSON object — no markdown fences, no extra text, no explanation.
