import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient


class DASPipelineQwen:

    def __init__(self,
                 openai_api_key,
                 student_model_id="Qwen/Qwen2.5-1.5B-Instruct"):

        print(f"Chargement du modèle étudiant : {student_model_id}")

        # ---------------- Student ----------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            student_model_id,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            student_model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.model.eval()

        # ---------------- Teacher ----------------
        self.client = InferenceClient(token=openai_api_key)
        self.teacher_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    # =====================================================
    # TEACHER
    # =====================================================

    def get_teacher_data(self, prompt, temperature=0.1, max_tokens=150):

        try:
            response = self.client.chat_completion(
                model=self.teacher_model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert NBA. Réponds uniquement en JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                logprobs=True
            )

            choice = response.choices[0]

            return {
                "content": choice.message.content,
                "logprobs": choice.logprobs
            }

        except Exception as e:
            raise RuntimeError(f"Teacher API error: {e}")

    # =====================================================
    # STUDENT LOGPROBS
    # =====================================================

    def get_student_logprobs(self, prompt, response):

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]

        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        token_losses = loss_fct(
            shift_logits.transpose(1, 2),
            shift_labels
        )

        token_logprobs = -token_losses
        valid_logprobs = token_logprobs.float().cpu().numpy().flatten()

        return {
            "mean_logprob": float(np.mean(valid_logprobs)),
            "total_logprob": float(np.sum(valid_logprobs))
        }

    # =====================================================
    # DECISION QUALITY SCORE
    # =====================================================

    def compute_quality_score(self, teacher_answer, student_answer):

        teacher_mean = 0.0

        if teacher_answer["logprobs"] is not None:
            try:
                teacher_mean = np.mean([
                    t.logprob
                    for t in teacher_answer["logprobs"]
                    if t.logprob is not None
                ])
            except:
                teacher_mean = 0.0

        student_mean = student_answer["mean_logprob"]

        divergence = teacher_mean - student_mean

        return 1 / (1 + np.exp(-divergence))

    # =====================================================
    # DATASET GENERATION
    # =====================================================

    def run(self,
        promptList,
        output_file="distillation_dataset.jsonl",
        quality_threshold=0.55):

        print(f"Nombre total de prompts : {len(promptList)}")

        # ⭐ Ouvrir en mode append (écriture progressive)
        with open(output_file, "w", encoding="utf-8") as f:

            i = 0

            for prompt in promptList:
                i += 1
                print(f"\n🔄 Traitement du prompt {i}/{len(promptList)}")

                try:
                    teacher_answer = self.get_teacher_data(prompt)

                    student_stats = self.get_student_logprobs(
                        prompt,
                        teacher_answer["content"]
                    )

                    quality_score = self.compute_quality_score(
                        teacher_answer,
                        student_stats
                    )

                    # ⭐ Filtrage qualité
                    if quality_score < quality_threshold:
                        continue

                    record = {
                        "text": f"""### Human:
                        {prompt}

                        ### Assistant:
                        {teacher_answer['content']}"""
                    }

                    # ⭐ Write immediately (streaming dataset build)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

                except Exception as e:
                    print(f"⚠️ Erreur sur prompt {i}: {e}")
                    continue

        print(f"✅ Dataset généré dans {output_file}")