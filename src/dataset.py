import pandas as pd
import requests
from datasets import load_dataset, Dataset
import json
import re
import numpy as np

# Emotions Datasets
class GoEmotionsDataset():
    """
    A dataset class for processing the GoEmotions dataset from the paper:
    "GoEmotions: A Dataset of Fine-Grained Emotions."
    
    This class retrieves and processes sentences annotated with fine-grained emotions
    from the GoEmotions dataset, filtering them based on a selected subset of emotions.
    
    Args:
        split (str): The dataset split to load. Options are 'train', 'valid', or 'test'.
        selected_emotions (list): A list of emotions to filter the dataset. Default includes
            ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust'].
    """
    def __init__(self,
                 split='train',
                 selected_emotions = ['sadness', 'joy', 'fear', 'anger', 'surprise', 'disgust']):
        self.split = split
        self.selected_emotions = selected_emotions
        
        # Retrieve the full list of emotions from the dataset metadata
        emotions_file_url = 'https://raw.githubusercontent.com/google-research/google-research/refs/heads/master/goemotions/data/emotions.txt'
        response = requests.get(emotions_file_url)
        self.emotions = response.text.split('\n')
    
    def get_base_dataset(self):
        """
        Loads and processes the GoEmotions dataset, filtering sentences based on the selected emotions.
        
        Returns:
            list[dict]: A list of dictionaries where each dictionary represents a sentence with
            its text ('sentence'), associated emotion ('style'), and an ID ('id').
        """
        ds = load_dataset("google-research-datasets/go_emotions", "simplified")
        
        dict_samples = {'sentence': [], 'labels': []}
        for i in range(ds[self.split].num_rows):
            dict_samples['sentence'].append(ds[self.split][i]['text'])
            dict_samples['labels'].append(ds[self.split][i]['labels'])
        
        metadata_df = pd.DataFrame(dict_samples)
        for i in range(len(self.emotions)):
            metadata_df[self.emotions[i]] = metadata_df.labels.apply(lambda x: i in x)
    
        # Get sentences with the selected emotions
        filtered_df = metadata_df[metadata_df[self.selected_emotions].any(axis=1)][['sentence']+self.selected_emotions]
    
        # Remove sentences with more than one emotion
        filtered_df = filtered_df[filtered_df[self.selected_emotions].sum(axis=1) <= 1]
    
        # Add "style" (emotion) column
        filtered_df['style'] = None
        for emotion in self.selected_emotions:
            filtered_df.loc[filtered_df[emotion], 'style'] = emotion
    
        # Use only id, sentence, and style columns
        filtered_df['id'] = filtered_df.index
        filtered_df = filtered_df[['id','sentence','style']]

        samples_list = filtered_df.to_dict('records')
        return samples_list

class StyleVectorsSentencesDataset():
    """
    A dataset class for processing sentences from the paper:
    "Style Vectors for Steering Generative Large Language Models."
    
    This class retrieves and processes factual and subjective sentences from the dataset
    available at:
    https://github.com/DLR-SC/style-vectors-for-steering-llms
    
    Attributes:
        sentences_url (dict): A dictionary containing URLs to factual and subjective sentences.
    """
    def __init__(self):
        self.sentences_url = {'factual': 'https://raw.githubusercontent.com/DLR-SC/style-vectors-for-steering-llms/refs/heads/main/evaluation_prompts/factual_sentences.txt',
                    'subjective': 'https://raw.githubusercontent.com/DLR-SC/style-vectors-for-steering-llms/refs/heads/main/evaluation_prompts/subjective_sentences.txt'
                    }
        
    def get_base_dataset(self):
        """
        Retrieves and processes sentences from the dataset, categorizing them as factual or subjective.
        
        Returns:
            list[dict]: A list of dictionaries where each dictionary represents a sentence with
            its text ('sentences'), style ('style' - either 'factual' or 'subjective'), and an ID ('id').
        """
        sentences_dfs = []
        for sentence_type in ['factual', 'subjective']:
            response = requests.get(self.sentences_url[sentence_type])
            sentences = response.text.split('\n')
            sentences_type_df = pd.DataFrame()
            sentences_type_df['sentence'] = sentences
            sentences_type_df['style'] = sentence_type
            sentences_type_df['id'] = sentences_type_df.index
            if sentence_type=='factual':
                sentences_type_df['id'] = sentences_type_df['id'].apply(lambda x: f'F{x+1:02d}')
            elif sentence_type=='subjective':
                sentences_type_df['id'] = sentences_type_df['id'].apply(lambda x: f'S{x+1:02d}')
            sentences_dfs.append(sentences_type_df)
            
        sentences_df = pd.concat(sentences_dfs)
        samples_list = sentences_df.to_dict('records')
        return samples_list

# AI Persona Dataset
class AdvancedAIRiskDataset:
    """
    Retrieve & process JSONL files from
    advanced-ai-risk/human_generated_evals (Anthropic-Evals).

    Parameters
    ----------
    dataset_dir : str
        Path whose last component ends with "_<file-stem>".
        Example: ".../foo_self-awareness-general-ai/".
    question_type : {"mcq", "qa"}, default "mcq"
        mcq - keep (A)/(B) lines; answers are letters.
        qa  - strip choice lines; answers are full strings.
    switch_answer_options : bool, default False
        Swap answer / alt_answer after processing.
    """

    def __init__(
        self,
        dataset_dir: str,
        question_type: str = "mcq",
        switch_answer_options: bool = False,
    ):
        self.base_url = (
            "https://raw.githubusercontent.com/anthropics/evals/"
            "refs/heads/main/advanced-ai-risk/human_generated_evals/"
        )
        self.dataset_names = self.get_all_dataset_names()
        self.switch_answer_options = switch_answer_options

        question_type = question_type.lower()
        if question_type not in {"mcq", "qa"}:
            raise ValueError("question_type must be 'mcq' or 'qa'")
        self.question_type = question_type

        tail = dataset_dir.rstrip("/").split("/")[-1]
        self.dataset = tail.split("_")[-1]
        if self.dataset not in self.dataset_names:
            raise ValueError(
                f"Dataset '{self.dataset}' not found. "
                f"Valid options are: {sorted(self.dataset_names)}"
            )

    def get_all_dataset_names(self) -> list[str]:
        url_git = (
            "https://api.github.com/repos/anthropics/evals/contents/"
            "advanced-ai-risk/human_generated_evals"
        )
        response = requests.get(url_git)
        response.raise_for_status()
        files = response.json()

        jsonl_files = [
            f for f in files if f['type'] == 'file' and f['name'].endswith('.jsonl')
        ]
        return [jf['name'].split('.')[0] for jf in jsonl_files]

    def get_base_dataset(self) -> list[dict]:
        dataset_url = f"{self.base_url}{self.dataset}.jsonl"
        response = requests.get(dataset_url, timeout=30)
        response.raise_for_status()
        df = pd.DataFrame(json.loads(l) for l in response.text.splitlines() if l.strip())

        # strip parens -> letters
        def _letter(s: str) -> str:
            return re.sub(r"[()\s]", "", s)[:1]

        df["ans_letter"]  = df["answer_not_matching_behavior"].map(_letter)
        df["alt_letter"]  = df["answer_matching_behavior"].map(_letter)

        if self.question_type == "qa":
            def _choices_map(q: str) -> dict:
                pat = r"\(\s*([A-Z])\s*\)\s*([^\n]+)"   # capture only to \n
                return {m[0]: m[1].strip() for m in re.findall(pat, q)}
            df["choice_map"] = df["question"].map(_choices_map)
            def _full_text(row, col):
                txt = row["choice_map"].get(row[col], row[col])
                return txt.split("\n", 1)[0].strip()    # ← NEW: trim at newline
            df["answer"]      = df.apply(lambda r: _full_text(r, "ans_letter"), axis=1)
            df["alt_answer"]  = df.apply(lambda r: _full_text(r, "alt_letter"), axis=1)
        
            def _clean_q(q: str) -> str:
                for kw in ("Choices:", "Answer:"):
                    if kw in q:
                        return q.split(kw, 1)[0].rstrip()
                m = re.search(r"\(\s*[A-Z]\s*\)", q)
                return q[:m.start()].rstrip() if m else q.strip()
            df["question_final"] = df["question"].map(_clean_q)

        else:  # mcq
            df["answer"], df["alt_answer"] = df["ans_letter"], df["alt_letter"]
            df["question_final"] = df["question"]

        # optional swap
        if self.switch_answer_options:
            df[["answer", "alt_answer"]] = df[["alt_answer", "answer"]]

        return (
            df[["question_final", "answer", "alt_answer"]]
            .rename(columns={"question_final": "question"})
            .to_dict("records")
        )

# QA Datasets
class TriviaQADataset():
    def __init__(
            self,
            split="train",
            dataset_dir_prefix="data",
            num_samples=5000
    ):
        self.dataset_id = "mandarjoshi/trivia_qa"
        self.split = split
        self.num_samples = num_samples

    def get_base_dataset(self):
        ds = load_dataset(self.dataset_id, name="rc.nocontext", split=self.split)
        if self.num_samples:
            ds = ds.select(range(min(self.num_samples, len(ds))))

        ids = []
        questions = []
        answers = []

        for i, sample in enumerate(ds):
            ids.append(str(i))
            questions.append(sample["question"])
            # TriviaQA has multiple possible answers, we take the first one
            answers.append(sample["answer"]["normalized_aliases"])

        data = Dataset.from_dict({"id": ids, "question": questions, "answer": answers, "alt_answer": [""] * len(answers)})
        samples_list = [sample for sample in data]
        return samples_list

class TruthfulQADataset():
    def __init__(self):
        self.url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/refs/heads/main/TruthfulQA.csv"
        self.data = self.load_data()

    def load_data(self):
        import pandas as pd
        # first line is the header
        df = pd.read_csv(self.url, quotechar='"', on_bad_lines='skip')
        print(f"Loaded {len(df)} samples")
        return df

    def get_base_dataset(self):
        samples_list = []
        for i, row in self.data.iterrows():
            sample = {
                "id": str(i),
                "question": row["Question"],
                "answer": row["Best Answer"],
                "alt_answer": row["Best Incorrect Answer"]
            }
            samples_list.append(sample)
        return samples_list

# Toxicity
class RealToxicityPromptsDataset:
    """
    Wrapper for "allenai/real-toxicity-prompts" dataset.
    """
    def __init__(self, split: str = "train", num_samples: int = None, shuffle: bool = False, seed: int = 42):
        """
        Args:
            split (str): The split of the dataset to load. Default is "train".
        """
        self.dataset = load_dataset("allenai/real-toxicity-prompts", split=split)
        self.dataset = self.dataset.shuffle(seed=seed) if shuffle else self.dataset
        samples = []
        num_samples = len(self.dataset) if num_samples is None else num_samples
        num_challenging_samples = num_samples // 2
        for sample in self.dataset:
            if sample['challenging']:
                samples.append(sample)
            if len(samples) >= num_challenging_samples:
                break
        # Add non-challenging samples
        for sample in self.dataset:
            if not sample['challenging']:
                samples.append(sample)
            if len(samples) >= num_samples:
                break
        self.dataset = samples

    def __len__(self):
        return len(self.dataset)
    
    def get_base_dataset(self):
        """
        Returns:
            list[dict]: A list of dictionaries where each dictionary represents a sample with
            its text ('question'), a toxic completion ('answer') and an ID ('id').
        """

        samples_list = []
        for i in range(len(self.dataset)):
            samples_list.append({
                'question': self.dataset[i]['prompt']['text'],
                'answer': self.dataset[i]['continuation']['text'],
                'alt_answer': None,
                'id': i
            })
        return samples_list
    