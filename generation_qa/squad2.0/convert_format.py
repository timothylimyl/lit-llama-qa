
from pathlib import Path
import json

train_path = Path("train-v2.0.json")
dev_path = Path("dev-v2.0.json")

# Open .json file
with open(train_path, "rb") as f:
    train_squad_dict = json.load(f)
with open(dev_path, "rb") as f:
    dev_squad_dict = json.load(f)


def convert_data_format(squad_dict,data_filename):
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
                with open(f"squad_data_{data_filename}.jsonl", "a") as file:
                    file.write(json.dumps(full_data_dict) + "\n")


    print(f"Total unanswerable questions: {unanswerable_count}")
    print(f"Total unique context: {total_unique_context}")
    print(f"Total dataset rows: {len(queries)}")


convert_data_format(train_squad_dict,"train")
convert_data_format(dev_squad_dict,"dev")