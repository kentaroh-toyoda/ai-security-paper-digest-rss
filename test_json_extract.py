import json


def test_json_extraction():
    test_response = '''◁think▷Okay, let's see. The paper is about a Semantic Irony Recognition Network (SemIRNet) for multimodal sarcasm detection. The main focus is improving the accuracy of detecting irony by using ConceptNet for knowledge, cross-modal semantic similarity modules, and contrastive learning.

First, I need to check if this relates to AI security, safety, or red teaming. The abstract mentions enhancing model performance for irony detection, which is an NLP task. But I don't see any direct mention of security, adversarial attacks, safety evaluations, or red teaming. The improvements are in the model's ability to detect sarcasm using multimodal data and knowledge bases, which is more about performance and accuracy rather than security or safety.

Looking at the relevance criteria, the paper doesn't fit into any of the security or safety categories. It's not about red teaming, adversarial prompting, data poisoning, or any of those. It's about improving a specific AI application. So, this paper is probably not relevant to AI security or safety.

Therefore, the answer should be that the paper is not relevant. The JSON response would have "relevant": false. Since it's not relevant, the other fields like summary, tags, etc., don't need to be filled out. But maybe the user expects some explanation. Wait, the instructions say if it's not relevant, just return {"relevant": false}. So that's the case here.
◁/think▷

```json
{
    "relevant": false
}
```'''

    print("Original response:")
    print(repr(test_response))
    print()

    # Try direct JSON parsing first
    try:
        result_dict = json.loads(test_response)
        print("✅ Direct JSON parsing succeeded")
        return result_dict
    except json.JSONDecodeError:
        print("❌ Direct JSON parsing failed")

    # Try to extract JSON from the response
    last_brace = test_response.rfind('{')
    if last_brace != -1:
        print(f"Found last brace at position: {last_brace}")

        # Find the matching closing brace
        brace_count = 0
        end_pos = last_brace
        for i in range(last_brace, len(test_response)):
            if test_response[i] == '{':
                brace_count += 1
            elif test_response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break

        if brace_count == 0:  # Found matching braces
            json_str = test_response[last_brace:end_pos]
            print(f"Extracted JSON string: {repr(json_str)}")

            try:
                result_dict = json.loads(json_str)
                print("✅ Successfully extracted and parsed JSON")
                return result_dict
            except json.JSONDecodeError:
                print(f"❌ Failed to parse extracted JSON: {json_str}")

    print("❌ All extraction methods failed")
    return {"relevant": False}


if __name__ == "__main__":
    result = test_json_extraction()
    print(f"Final result: {result}")
