from pathlib import Path
import json

# Give the path for train data
path = Path("squad2.0/dev-v2.0.json")

# Open .json file
with open(path, "rb") as f:
    squad_dict = json.load(f)

contexts = []
queries = []
answers = []
unanswerable_count = 0
total_unique_context = 0
# Search for each passage, its question and its answer
for group in squad_dict["data"]:
    for passage in group["paragraphs"]:
        context = passage["context"]
        total_unique_context += 1
        for qa in passage["qas"]:
            question = qa["question"]
            try:
                answer = qa["answers"][0]["text"]  # just take the first answer
            except:
                if qa["answers"] == []:  # unanswerable
                    answer = "<unk>"
                    unanswerable_count += 1
                else:
                    SystemExit(f"Unexpected answer format, answer: f{qa['answers']}")

            contexts.append(context)
            queries.append(question)
            answers.append(answer)

            full_data_dict = {
                "context": context,
                "question": question,
                "answer": answer,
            }
            with open("squad_data_dev.jsonl", "a") as file:
                file.write(json.dumps(full_data_dict) + "\n")


print(f"Total unanswerable questions: {unanswerable_count}")
print(f"Total unique context: {total_unique_context}")
print(f"Total dataset rows: {len(queries)}")
