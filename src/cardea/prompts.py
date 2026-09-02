"""Shared prompt text and structured-output formatting."""

import json

# Prepended to the study-level prompts so the model localizes findings with
# bounding boxes in its reasoning. Keep verbatim.
GROUNDING_PROMPT = (
    "You better localize features using bounding boxes (formatted as JSON with "
    "img_idx, bbox_2d, and label) to explicitly guide your visual attention in reasoning process."
)

_FORMAT_PREAMBLE = '''STRICT OUTPUT FORMAT:
- Return only the JSON value that conforms to the schema. Do not include any additional text, explanations, headings, or separators.
- Do not wrap the JSON in Markdown or code fences (no ``` or ```json).
- Do not prepend or append any text (e.g., do not write "Here is the JSON:").
- The response must be a single top-level JSON value exactly as required by the schema (object/array/etc.), with no trailing commas or comments.

The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]} the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

Here is the output schema (shown in a code block for readability only — do not include any backticks or Markdown in your output):'''


def format_instructions(model):
    schema = dict(model.model_json_schema())
    schema.pop("title", None)
    schema.pop("type", None)
    return (
        _FORMAT_PREAMBLE
        + "\n```\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\n```"
    )
