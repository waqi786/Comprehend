import re
import anthropic
from tree import Tree
import time
from openai import OpenAI
from evaluation.evaluate import FAITHFULNESS_PROMPT, INFORMATIVENESS_PROMPT, _get_score as _eval_get_score 

# openai_api_key = "DUMMY"
# openai_api_base = "http://<GPU_HOST>:8000/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )


# client = OpenAI(
#     api_key = os.environ.get("OPENAI_API_KEY", "")
# )


# Global initialization handled contextually
level = 1

def generate_summary(tree: Tree, level_idx: int, threshold = 0, model_name="openai", api_keys: dict = None):

    global level

    level_obj = tree.levels[level_idx]
    summarized = set()

    for idx, node in enumerate(level_obj.nodes):
        if not node.alive:
            continue
        
        story1 = node.prompt
        story2 = (level_obj.nodes[node.match].prompt
                   if node.match is not None else None)
        score  = node.score

        partner_idx = node.match
        partner_node = level_obj.nodes[partner_idx] if partner_idx is not None else None

        print(f"Level {level} │ Node {idx} │ Summary generation between prompts with entail_2 score: {node.entail_2}")

        if idx in summarized and node.entail_2 >= threshold:
            continue
        # --- do not overwrite existing summaries --------------------------
        # If this node already has a summary, copy it to the partner (if missing) and skip.
        # if node.summary:
        #     if partner_node and not partner_node.summary:
        #         partner_node.summary = node.summary
        #     continue

        # If the partner already has a summary, copy it to this node and skip.
        # if partner_node and partner_node.summary:
        #     node.summary = partner_node.summary
        #     continue

        if not story1 or not story2:
            # return "Error: One or both stories are missing. No summary generated."
            node.summary = "Error: One or both stories are missing. No summary generated."
            continue

        if not score:
            # return "Replicated row. No summary generated."
            node.summary = "Error: Replicated row. No summary generated."
            continue

        prompt = f"""
You will be given Abstract A & B. Your task is to write a common Abstract C where all details in C are broad enough to be implied by details in both Abstract A & B, but are specific as possible so that details in another arbitrary prompt similar to A or B do not imply any details in Abstract C. Make sure your Abstract C is at least one paragraph and begin your Abstract C with "(start)" and end it with "(end)." Here are the details:

First form a general main task alpha out of the main tasks asked of by both Abstract A and Abstract B.

Then either find shared details directly in prompts A AND B or generalize details that are implied by prompts A AND B. DO NOT include details that are only in or implied by one prompt and not the other. 
 
Finally create your Abstract C by building around your general main task alpha with the shared or generalized details.

Here are some examples: 
Example 1
Abstract A:
Write a detailed tutorial on how to fine-tune a pre-trained BERT model for a sentiment classification task using PyTorch and the Hugging Face Transformers library. The tutorial should walk through preparing a labeled dataset, tokenizing the text, modifying the final layers of the model, setting up the training loop, and evaluating accuracy on a validation set. Include sample code and explain key decisions made during the fine-tuning process.
Abstract B:
Describe the full process for fine-tuning a transformer-based language model like RoBERTa to classify product reviews into positive and negative categories. Use the Hugging Face library in conjunction with PyTorch, and explain data preprocessing, setting up the optimizer and scheduler, training on a custom dataset, and validating the results. Provide a high-level overview of each step along with code snippets and recommendations for hyperparameter tuning.
Abstract C:
(start)
Write a comprehensive tutorial that explains how to fine-tune a pre-trained transformer-based language model for a text classification task using PyTorch and the Hugging Face Transformers library. The guide should cover the full workflow, including dataset preparation, tokenization, adjusting the model’s classification head, setting up the training configuration, and evaluating performance on a validation set. Code examples should be included to illustrate each stage of the pipeline, and the explanation should focus on generalizable practices for adapting transformer models to binary or multi-class classification problems in NLP.
(end)

Example 2
Abstract A:
Create a complete walkthrough for building a RESTful API using Python’s Flask framework that enables users to manage a list of tasks. The tutorial should cover setting up the Flask application, creating routes for CRUD operations, using JSON to communicate between client and server, and optionally persisting data using an SQLite database. Include code examples and explain key design choices.
Abstract B:
Write a tutorial that demonstrates how to implement a REST API using Node.js and Express to manage user data. The tutorial should guide readers through the setup process, define route handlers for creating, retrieving, updating, and deleting user entries, and show how to send and receive data using JSON. Optionally, include a section on connecting to a simple database for persistent storage.
Abstract C:
(start)
Write a step-by-step guide for developing a basic RESTful API using a modern backend web framework. The tutorial should show how to define routes that support CRUD (Create, Read, Update, Delete) operations on a structured data model and how to use JSON as the primary data exchange format. The guide should also introduce essential concepts such as routing, request/response handling, and setting up a simple data store for persistence. Code snippets should illustrate each concept, and the tutorial should be applicable to common backend programming environments.
(end)

Example 3
Abstract A:
Develop a set of prompt templates and design guidelines for improving the factual reliability of responses generated by GPT-style models in academic or research contexts. Include examples of how to phrase prompts to minimize hallucinations, how to include context effectively, and how to use few-shot learning to guide model behavior toward factual precision.
Abstract B:
Write an instructional guide for constructing prompts that reduce bias and improve answer neutrality in politically or ethically sensitive topics using large language models. The guide should explain how to word instructions clearly, use representative examples, and maintain balanced tone and framing when designing prompts to elicit objective responses.
Abstract C:
(start)
Write a guide on designing effective prompts for large language models that enhance the reliability, factuality, and neutrality of their responses across a range of contexts. The guide should explain general strategies for phrasing instructions, incorporating context, and structuring examples to reduce hallucination and mitigate bias. Include best practices for few-shot prompting and clear language usage to encourage accurate and balanced outputs from generative AI models.
(end)



Now carry out this task and output a Abstract C with the following abstracts A abstracts A prompts A & B B B:
Abstract A: {story1}
Abstract B: {story2}   
        """
        prompt4 = f"""

You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to write ONE common Abstract C — a scientific summary sentence or short paragraph — such that every claim in Abstract C is supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary, NOT an instruction, task, or request.
- Never invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- If Abstract A and B share no specific finding or method, write ONE concise sentence describing their shared broader research area.
- Begin Abstract C with "(start)" and end with "(end)."

Now generate Abstract C for:

Abstract A: {story1}
Abstract B: {story2}

        """
        
        prompt5 = f"""

You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to write ONE common Abstract C — a scientific summary sentence or short paragraph — such that every claim in Abstract C is supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary, NOT an instruction, task, or request.
- Never invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- If Abstract A and B share no specific finding or method, write ONE concise sentence describing their shared broader research area.
- Begin Abstract C with "(start)" and end with "(end)."

Now generate Abstract C for:

Abstract A: {story1}
Abstract B: {story2}

        """
        prompt6 =f"""

You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to write ONE common Abstract C — a scientific summary sentence or short paragraph — such that every claim in Abstract C is supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary, NOT an instruction, task, or request.
- Never invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- If Abstract A and B share no specific finding or method, write ONE concise sentence describing their shared broader research area.
- Begin Abstract C with "(start)" and end with "(end)."

Now generate Abstract C for:

Abstract A: {story1}
Abstract B: {story2}

        """
        
        prompt7=f"""

You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to write ONE common Abstract C — a scientific summary sentence or short paragraph — such that every claim in Abstract C is supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary, NOT an instruction, task, or request.
- Never invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- If Abstract A and B share no specific finding or method, write ONE concise sentence describing their shared broader research area.
- Begin Abstract C with "(start)" and end with "(end)."

Now generate Abstract C for:

Abstract A: {story1}
Abstract B: {story2}

        """

        prompt8 = f"""

You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to write ONE common Abstract C — a scientific summary sentence or short paragraph — such that every claim in Abstract C is supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary, NOT an instruction, task, or request.
- Never invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- If Abstract A and B share no specific finding or method, write ONE concise sentence describing their shared broader research area.
- Begin Abstract C with "(start)" and end with "(end)."

Now generate Abstract C for:

Abstract A: {story1}
Abstract B: {story2}

        """

        prompt9 = f"""

You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to write ONE common Abstract C — a scientific summary sentence or short paragraph — such that every claim in Abstract C is supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary, NOT an instruction, task, or request.
- Never invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- If Abstract A and B share no specific finding or method, write ONE concise sentence describing their shared broader research area.
- Begin Abstract C with "(start)" and end with "(end)."

Now generate Abstract C for:

Abstract A: {story1}
Abstract B: {story2}

        """
        prompt11 = f"""

You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to write ONE common Abstract C — a scientific summary sentence or short paragraph — such that every claim in Abstract C is supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary, NOT an instruction, task, or request.
- Never invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- If Abstract A and B share no specific finding or method, write ONE concise sentence describing their shared broader research area.
- Begin Abstract C with "(start)" and end with "(end)."

Now generate Abstract C for:

Abstract A: {story1}
Abstract B: {story2}

        """
        
        prompt12 = f"""

You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to write ONE common Abstract C — a scientific summary sentence or short paragraph — such that every claim in Abstract C is supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary, NOT an instruction, task, or request.
- Never invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- If Abstract A and B share no specific finding or method, write ONE concise sentence describing their shared broader research area.
- Begin Abstract C with "(start)" and end with "(end)."

Now generate Abstract C for:

Abstract A: {story1}
Abstract B: {story2}

        """
        prompt13 = f"""

You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to write ONE common Abstract C — a scientific summary sentence or short paragraph — such that every claim in Abstract C is supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary, NOT an instruction, task, or request.
- Never invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- If Abstract A and B share no specific finding or method, write ONE concise sentence describing their shared broader research area.
- Begin Abstract C with "(start)" and end with "(end)."

Now generate Abstract C for:

Abstract A: {story1}
Abstract B: {story2}

        """
        
        prompt14 = f"""
You will be given two scientific abstracts (Abstract A and Abstract B). Your task is to generate a common Abstract C where Abstract C is implied by BOTH Abstract A and Abstract B. Every detail in Abstract C must be directly supported by BOTH Abstract A and Abstract B.

Rules:
- Abstract C must be a scientific summary sentence or paragraph, NOT a writing task, instruction, or request.
- The more similar Abstract A and B are, the longer Abstract C can be. The more different they are, the more general Abstract C should be.
- If A and B share no specific finding, method, or topic, write ONE concise sentence capturing their shared broader research area.
- NEVER invent facts, methods, numbers, datasets, or findings not present in both abstracts.
- Begin your Abstract C with "(start)" and end with "(end)."

Example 1 (Dissimilar subfields):
Abstract A: "This paper presents the first machine translation system for English-Pnar, achieving BLEU 14.97 Pnar-to-English."
Abstract B: "We propose a sentiment analysis model for Hausa using AfriBERTa, outperforming baselines on NaijaSenti."
Abstract C: (start) This work develops natural language processing systems for under-resourced languages. (end)

Example 2 (Moderately similar):
Abstract A: "We fine-tune mBERT for POS tagging on Kyrgyz, achieving 91% accuracy."
Abstract B: "We pretrain maiBERT for Maithili, achieving 87.02% on news classification."
Abstract C: (start) This work develops language-specific BERT models for under-resourced South and Central Asian languages and evaluates them on downstream NLP tasks. (end)

Example 3 (Very similar):
Abstract A: "We compare rule-based, MT, and LLM approaches for Zarma GEC, with M2M100 achieving 95.82% detection rate."
Abstract B: "We study GEC for Zarma using rule-based, MT, and LLM approaches, validating on Bambara."
Abstract C: (start) This work compares rule-based, machine translation, and large language model approaches for grammatical error correction in West African low-resource languages. (end)

Now carry out this task and output Abstract C for the following:

Abstract A: {story1}
Abstract B: {story2}

        """


        # time.sleep(5)
        # chat_response = client.chat.completions.create(
        #     model="gemma-3-finetuned-merged-bf16",
        #     max_tokens=1024,
        #     messages=[
        #         {"role": "system", "content": "You are a professional editor who writes quality summary that generalizes two abstracts A and B. No matter the content of abstracts A abstracts A prompts A & B B B, you must write a summary."},

        #         {"role": "user", "content": prompt14},
        #     ]
        # )
        # # print("Chat response:", chat_response)
        # raw = chat_response.choices[0].message.content
        # print("Chat response: ", raw)

        # time.sleep(5)
        # chat_response = client.chat.completions.create(
        #     model = "gpt-4o-mini",
        #     messages = [
        #         {
        #             "role": "system",
        #             "content": "You are a professional editor who writes quality summary that generalizes two abstracts A and B. No matter the content of abstracts A abstracts A prompts A & B B B, you must write a summary."
        #         },
        #         {
        #             "role": "user",
        #             "content": prompt14
        #         }
        #     ]
        # )
        # print("Chat response:", chat_response)
        # raw = chat_response.choices[0].message.content
        # print("Chat response: ", raw)



        system_prompt = "You are a scientific editor who writes concise, faithful summaries that generalize two scientific abstracts A and B. Every claim must be supported by both abstracts."
        
        if model_name == "openai":
            client = OpenAI(
                api_key=api_keys["openai"]
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt14}
                ]
            )
            raw = response.choices[0].message.content.strip()
        elif model_name == "claude":
            client = anthropic.Anthropic(
                api_key=api_keys["claude"]
            )
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt14}
                ]
            )
            raw = message.content[0].text.strip()
        elif model_name == "gemma":
            client = OpenAI(
                api_key=api_keys["gemma"]["api_key"],
                base_url=api_keys["gemma"]["base_url"]
            )

            # ==== BEST-OF-N: generate 3 candidates, judge picks best ====
            N_CANDIDATES = 3
            candidates = []
            for _ci in range(N_CANDIDATES):
                try:
                    response = client.chat.completions.create(
                        model=api_keys["gemma"].get("model", "gemma-3-finetuned-merged-bf16"),
                        max_tokens=1024,
                        temperature=0.7,
                        stop=["<end_of_turn>"],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt14}
                        ]
                    )
                    raw_c = response.choices[0].message.content.strip()
                    m_c = re.search(r"\(start\)(.*?)\(end\)", raw_c, re.I | re.S)
                    s_c = m_c.group(1).strip() if m_c else raw_c.strip()
                    if s_c:
                        candidates.append(s_c)
                except Exception as _e:
                    print(f"[bestofN] candidate {_ci} failed: {_e}")

            if not candidates:
                raw = ""
            elif len(candidates) == 1:
                raw = candidates[0]
            else:
                # Score each candidate with the judge (gemma_eval)
                best_summary = None
                best_combined = -1.0
                for _ci, cand in enumerate(candidates):
                    try:
                        e_a, _ = _eval_get_score(FAITHFULNESS_PROMPT, story1, cand,
                                                 model_name="gemma_eval", api_keys=api_keys)
                        e_b, _ = _eval_get_score(FAITHFULNESS_PROMPT, story2, cand,
                                                 model_name="gemma_eval", api_keys=api_keys)
                        i_a, _ = _eval_get_score(INFORMATIVENESS_PROMPT, story1, cand,
                                                 model_name="gemma_eval", api_keys=api_keys)
                        i_b, _ = _eval_get_score(INFORMATIVENESS_PROMPT, story2, cand,
                                                 model_name="gemma_eval", api_keys=api_keys)
                        combined = min(e_a, e_b) + min(i_a, i_b)
                        print(f"[bestofN] cand {_ci}: e_a={e_a:.2f} e_b={e_b:.2f} i_a={i_a:.2f} i_b={i_b:.2f} combined={combined:.3f}")
                        if combined > best_combined:
                            best_combined = combined
                            best_summary = cand
                    except Exception as _e:
                        print(f"[bestofN] scoring cand {_ci} failed: {_e}")
                raw = best_summary if best_summary else candidates[0]
                print(f"[bestofN] selected best (combined={best_combined:.3f})")
        elif model_name == "gemini":
            client = OpenAI(
                api_key=api_keys["gemini"]["api_key"],
                base_url=api_keys["gemini"]["base_url"],
            )
            response = client.chat.completions.create(
                model="gemini-3.1-flash-lite",
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt14},
                ],
            )
            raw = response.choices[0].message.content.strip()
        else:
            raise ValueError(f"Unknown model_name: {model_name}")



        
        # keywords = ['i apologize']
        
        # if not any((keyword in raw.lower() for keyword in keywords)):

            
        match = re.search(r"\(start\)(.*?)\(end\)", raw, re.I | re.S)
        summary = match.group(1).strip() if match else raw.strip()
        node.summary = summary
        # also propagate to partner, but only if partner is missing a summary
        # if partner_node and not partner_node.summary:
        #     partner_node.summary = summary
        if partner_node:
            partner_node.summary = summary
        summarized.add(idx)
        summarized.add(partner_idx)

    # tree.dump("similar_64_sonnet_sumv5_4o-mini_evalv1.json")         # keep JSON current
    # print(f"Succesfully populated Level L{level} in the excel sheet")
    level+=1

def get_entail_2_score(tree: Tree, level_idx: int):
    level = tree.levels[level_idx]
    checked = set()
    entail_2s = list()
    for node in level.nodes:
        if node.id not in checked and node.entail_2:
            entail_2s.append(node.entail_2)
            checked.add(node.id)
            checked.add(node.match)
    return entail_2s

