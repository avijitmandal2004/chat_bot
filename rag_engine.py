"""
DataChat RAG Engine — v6 (Self-Routing)
========================================
The LLM decides what to do — no pattern matching needed.

Flow:
  1. LLM sees the question + dataset schema
  2. LLM decides: "needs_code" or "needs_explanation"
  3. If needs_code  → LLM writes Pandas code → Python runs it → exact answer
  4. If needs_explanation → LLM explains using schema knowledge
  
This handles ANY question correctly without hardcoded patterns.
"""

import os, json, re
import pandas as pd
import numpy as np
import faiss
import pdfplumber
import chardet
import torch
from sentence_transformers import SentenceTransformer


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  🎮 GPU: {name} ({vram:.1f} GB VRAM) → CUDA")
        return 'cuda'
    print("  💻 No GPU → CPU")
    return 'cpu'


class RAGEngine:

    def __init__(self):
        self.device = get_device()
        print("⏳ Loading embedding model…")
        self.embedder     = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.dim          = 384
        self.index        = None
        self.chunks       = []
        self.doc_meta     = {}
        self._loaded      = False
        self._df          = None
        self._df_schema   = ''
        self._hf_pipeline = None
        print(f"✅ Ready on {self.device.upper()}.\n")

    # ── Document loading ──────────────────────────────────────
    def load_document(self, filepath, filename):
        ext = filename.rsplit('.', 1)[-1].lower()
        print(f"📂 Loading {filename} …")
        loaders = {'csv': self._load_csv, 'xlsx': self._load_xlsx,
                   'pdf': self._load_pdf,  'txt':  self._load_txt,
                   'json': self._load_json}
        if ext not in loaders:
            raise ValueError(f"Unsupported: {ext}")
        chunks, meta = loaders[ext](filepath, filename)
        self._build_index(chunks)
        self.doc_meta = meta
        self._loaded  = True
        print(f"✅ Indexed {len(chunks)} chunks.\n")
        return {**meta, 'chunks': len(chunks), 'status': 'ok'}

    def _load_csv(self, path, name):
        df = pd.read_csv(path)
        for col in df.columns:
            if any(x in col.lower() for x in ['date','time','month','year']):
                try: df[col] = pd.to_datetime(df[col], errors='coerce')
                except: pass
        self._df = df
        self._df_schema = self._build_schema(df, name)
        meta = {'filename': name, 'type': 'csv', 'rows': len(df),
                'columns': list(df.columns),
                'preview': df.head(5).to_dict(orient='records')}
        return self._df_to_chunks(df, name), meta

    def _load_xlsx(self, path, name):
        df = pd.read_excel(path)
        self._df = df
        self._df_schema = self._build_schema(df, name)
        meta = {'filename': name, 'type': 'xlsx', 'rows': len(df),
                'columns': list(df.columns),
                'preview': df.head(5).to_dict(orient='records')}
        return self._df_to_chunks(df, name), meta

    def _build_schema(self, df, name):
        lines = [
            f"DataFrame: df  |  File: {name}  |  {len(df)} rows × {len(df.columns)} columns",
            "",
            "Columns:"
        ]
        for col in df.columns:
            dtype   = str(df[col].dtype)
            sample  = df[col].dropna().head(3).tolist()
            sample_str = ', '.join(str(v) for v in sample)
            lines.append(f"  '{col}' ({dtype}) — e.g. {sample_str}")

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            lines.append("\nUnique values per categorical column:")
            for col in cat_cols[:10]:
                uv = df[col].dropna().unique().tolist()
                uv_str = ', '.join(f"'{v}'" for v in uv[:20])
                lines.append(f"  '{col}': [{uv_str}{'...' if len(uv)>20 else ''}]")

        num_cols = df.select_dtypes(include='number').columns.tolist()
        if num_cols:
            lines.append("\nNumeric column ranges:")
            for col in num_cols:
                lines.append(f"  '{col}': min={df[col].min():,.2f}, max={df[col].max():,.2f}, mean={df[col].mean():,.2f}")

        return '\n'.join(lines)

    def _df_to_chunks(self, df, name):
        chunks = [self._df_schema]
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if num_cols:
            chunks.append(f"Stats:\n{df[num_cols].describe().round(2).to_string()}")
        for col in cat_cols[:8]:
            chunks.append(f"Value counts '{col}':\n{df[col].value_counts().head(15).to_string()}")
        for i in range(0, len(df), 30):
            chunks.append(df.iloc[i:i+30].to_string(index=False))
        return chunks

    def _load_pdf(self, path, name):
        self._df = None; self._df_schema = ''
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, p in enumerate(pdf.pages):
                t = p.extract_text() or ''
                if t.strip(): pages.append(f"[Page {i+1}]\n{t}")
        full = '\n\n'.join(pages)
        return self._split_text(full), {'filename': name, 'type': 'pdf',
                                        'pages': len(pages), 'preview': full[:500]}

    def _load_txt(self, path, name):
        self._df = None; self._df_schema = ''
        with open(path, 'rb') as f: raw = f.read()
        enc  = chardet.detect(raw)['encoding'] or 'utf-8'
        text = raw.decode(enc, errors='replace')
        return self._split_text(text), {'filename': name, 'type': 'txt',
                                        'chars': len(text), 'preview': text[:500]}

    def _load_json(self, path, name):
        with open(path) as f: data = json.load(f)
        if isinstance(data, list):
            try:
                df = pd.json_normalize(data); self._df = df
                self._df_schema = self._build_schema(df, name)
                return self._df_to_chunks(df, name), {
                    'filename': name, 'type': 'json',
                    'rows': len(df), 'columns': list(df.columns),
                    'preview': df.head(5).to_dict(orient='records')}
            except: pass
        self._df = None; self._df_schema = ''
        text = json.dumps(data, indent=2)
        return self._split_text(text), {'filename': name, 'type': 'json',
                                        'preview': text[:500]}

    def _split_text(self, text, chunk_size=400, overlap=60):
        words = text.split()
        out = []
        for i in range(0, len(words), chunk_size - overlap):
            c = ' '.join(words[i:i+chunk_size])
            if c.strip(): out.append(c)
        return out or [text[:chunk_size]]

    # ── FAISS ─────────────────────────────────────────────────
    def _build_index(self, chunks):
        self.chunks = chunks
        emb = self.embedder.encode(chunks, show_progress_bar=False, batch_size=64)
        emb = np.array(emb, dtype='float32')
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)

    def _retrieve(self, query, top_k=5):
        q = np.array(self.embedder.encode([query]), dtype='float32')
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, top_k)
        return [(float(s), self.chunks[i]) for s, i in zip(scores[0], idxs[0]) if i >= 0]

    # ── Safe code execution ───────────────────────────────────
    def _execute_code(self, code, df):
        safe_builtins = {
            'len': len, 'sum': sum, 'min': min, 'max': max,
            'round': round, 'abs': abs, 'int': int, 'float': float,
            'str': str, 'bool': bool, 'list': list, 'dict': dict,
            'range': range, 'enumerate': enumerate, 'zip': zip,
            'sorted': sorted, 'print': print, 'isinstance': isinstance,
            'type': type, 'any': any, 'all': all,
        }
        local_vars  = {'df': df, 'pd': pd, 'np': np}
        exec_globals = {'__builtins__': safe_builtins}
        try:
            exec(code, exec_globals, local_vars)
            result = local_vars.get('result', None)
            if result is None:
                lines = [l.strip() for l in code.strip().split('\n') if l.strip()]
                if lines:
                    result = eval(lines[-1], exec_globals, local_vars)
            return result, None
        except Exception as e:
            return None, str(e)

    def _result_to_str(self, result):
        if result is None: return "No result"
        if isinstance(result, pd.DataFrame):
            return (result.head(30).to_string() +
                    (f"\n... ({len(result)} rows total)" if len(result) > 30 else ""))
        if isinstance(result, pd.Series):
            return (result.head(30).to_string() +
                    (f"\n... ({len(result)} items)" if len(result) > 30 else ""))
        if isinstance(result, (int, np.integer)):
            return f"{result:,}"
        if isinstance(result, (float, np.floating)):
            return f"{result:,.4f}"
        if isinstance(result, list):
            preview = result[:50]
            return str(preview) + (f"... ({len(result)} total)" if len(result) > 50 else "")
        return str(result)

    # ── MAIN ASK — Self-routing ───────────────────────────────
    def ask(self, question, backend='ollama'):
        """
        Single smart prompt that handles EVERYTHING:
        - The LLM decides if code is needed or not
        - If code: executes it and explains the result
        - If no code: explains directly from schema knowledge
        No hardcoded patterns. Works for any question.
        """
        if self._df is not None:
            return self._ask_smart(question, backend)
        return self._ask_rag(question, backend)

    def _ask_smart(self, question, backend):
        """
        One prompt does it all:
        LLM writes either CODE or DIRECT ANSWER based on the question type.
        """
        prompt = f"""You are a data analyst assistant. A dataset is loaded as `df`.

SCHEMA:
{self._df_schema}

STRICT RULES:
1. Any question with a NUMBER answer → must use CODE. NEVER guess numbers.
2. DIRECT only for: greetings, column definitions, general explanations (no numbers).
3. If your DIRECT answer would contain a number → use CODE instead.

FORMAT — pick one:

CODE:
result = <pandas expression>
EXPLAIN: <one line>

or

DIRECT: <plain English, zero numbers>

EXAMPLES:
"how many female" → CODE:\nresult = len(df[df['Gender']=='Female'])\nEXPLAIN: Count of females
"how many female married" → CODE:\nresult = len(df[(df['Gender']=='Female') & (df['Married']=='Yes')])\nEXPLAIN: Married females count
"average income" → CODE:\nresult = df['ApplicantIncome'].mean().round(2)\nEXPLAIN: Average income
"top 5 income" → CODE:\nresult = df.nlargest(5,'ApplicantIncome')[['Loan_ID','ApplicantIncome']]\nEXPLAIN: Top 5
"total loan" → CODE:\nresult = df['LoanAmount'].sum()\nEXPLAIN: Total loan amount
"correlation income loan" → CODE:\nresult = df[['ApplicantIncome','LoanAmount']].corr()\nEXPLAIN: Correlation
"hi" → DIRECT: Hello! Ask me anything about your dataset.
"what is loan status" → DIRECT: Loan_Status indicates approval — Y means approved, N means rejected.
"what does gender mean" → DIRECT: The Gender column records the applicant gender as Male or Female.
"tell me about dataset" → DIRECT: This is a loan application dataset with {len(self._df)} rows and {len(self._df.columns)} columns covering applicant demographics and loan details.

QUESTION: {question}
RESPONSE:"""
        raw = self._call_llm(prompt, backend, max_tokens=300)
        print(f"  🤖 LLM response: {raw[:200]}")

        # Parse CODE path
        if 'CODE:' in raw:
            # Extract code block
            code_match = re.search(r'CODE:\s*(.*?)(?:EXPLAIN:|$)', raw, re.DOTALL)
            explain_match = re.search(r'EXPLAIN:\s*(.*?)$', raw, re.DOTALL)

            if code_match:
                code = code_match.group(1).strip()
                # Clean up markdown
                code = re.sub(r'```python|```', '', code).strip()
                print(f"  📝 Code: {code}")

                result, error = self._execute_code(code, self._df)

                if error:
                    print(f"  ⚠️ Error: {error} — retrying…")
                    # Send error back to LLM to fix
                    fix_prompt = f"""This Pandas code failed:
Code: {code}
Error: {error}

Dataset schema:
{self._df_schema}

Write corrected code only (result = ...):"""
                    fixed_code = self._call_llm(fix_prompt, backend, max_tokens=150)
                    fixed_code = re.sub(r'```python|```|CODE:|EXPLAIN:.*', '', fixed_code).strip()
                    result, error2 = self._execute_code(fixed_code, self._df)
                    if error2:
                        print(f"  ⚠️ Retry failed: {error2}")
                        return self._ask_rag(question, backend)

                result_str = self._result_to_str(result)
                print(f"  ✅ Result: {result_str[:100]}")

                hint = explain_match.group(1).strip() if explain_match else ''

                # If result is a DataFrame/Series — return formatted directly, skip LLM
                if isinstance(result, (pd.DataFrame, pd.Series)):
                    rows = len(result)
                    header = f"Here are the results ({rows} {'row' if rows==1 else 'rows'}):\n\n"
                    return header + result_str

                # If result is a simple scalar — return with brief explanation
                if isinstance(result, (int, float, np.integer, np.floating)):
                    explain_prompt = f"""A user asked: "{question}"
The exact answer computed from the data is: {result_str}
{f'({hint})' if hint else ''}
Write ONE clear sentence stating this answer. Include the number. No extra text."""
                    return self._call_llm(explain_prompt, backend, max_tokens=80)

                # For lists and other types — brief LLM explanation
                explain_prompt = f"""A user asked: "{question}"

Exact result from the dataset:
{result_str}

{f'Context: {hint}' if hint else ''}

Give a clear, concise answer using these exact values. No disclaimers."""
                return self._call_llm(explain_prompt, backend, max_tokens=200)

        # Parse DIRECT path
        if 'DIRECT:' in raw:
            direct_match = re.search(r'DIRECT:\s*(.*?)$', raw, re.DOTALL)
            if direct_match:
                return direct_match.group(1).strip()

        # Fallback — return raw response if neither tag found
        # (sometimes LLM answers directly without tags)
        cleaned = re.sub(r'CODE:|DIRECT:|EXPLAIN:', '', raw).strip()
        return cleaned if cleaned else self._ask_rag(question, backend)

    def _ask_rag(self, question, backend):
        """RAG path for PDF/text files."""
        hits    = self._retrieve(question, top_k=5)
        context = '\n\n---\n\n'.join(c for _, c in hits)
        if self._df_schema:
            context = self._df_schema + '\n\n---\n\n' + context
        prompt = (f"You are a helpful data analyst. Answer using ONLY the context.\n"
                  f"Be concise and direct.\n\nCONTEXT:\n{context}\n\n"
                  f"QUESTION: {question}\n\nANSWER:")
        return self._call_llm(prompt, backend, max_tokens=350)

    # ── LLM caller ────────────────────────────────────────────
    def _call_llm(self, prompt, backend, max_tokens=350):
        if backend == 'ollama':      return self._ask_ollama(prompt, max_tokens)
        if backend == 'huggingface': return self._ask_hf(prompt, max_tokens)
        raise ValueError(f"Unknown: {backend}")

    def _ask_ollama(self, prompt, max_tokens=350):
        import requests
        def get_models():
            try: return [m['name'] for m in
                         requests.get('http://localhost:11434/api/tags', timeout=5)
                         .json().get('models', [])]
            except: return []
        available = get_models()
        preferred = ['codellama:7b','mistral','mistral:latest','llama3.2','llama3.2:latest',
                     'llama3','gemma2:2b','phi3:mini','tinyllama']
        model = next((p for p in preferred if p in available),
                     available[0] if available else 'mistral')
        print(f"  🦙 {model}")
        try:
            resp = requests.post('http://localhost:11434/api/generate',
                json={'model': model, 'prompt': prompt, 'stream': False,
                      'options': {'temperature': 0.1, 'num_predict': max_tokens, 'num_gpu': 99}},
                timeout=300)
            resp.raise_for_status()
            return resp.json().get('response', '').strip() or "❌ Empty response."
        except requests.exceptions.ConnectionError:
            return "❌ Ollama not running. Run: ollama serve"
        except requests.exceptions.Timeout:
            return "❌ Timed out. Wait 30s and retry."
        except Exception as e:
            return f"❌ Error: {e}"

    def _ask_hf(self, prompt, max_tokens=350):
        try:
            if self._hf_pipeline is None:
                from transformers import pipeline
                model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
                print(f"  ⏳ Loading {model_id}…")
                self._hf_pipeline = pipeline('text-generation', model=model_id,
                    device_map='auto',
                    torch_dtype=torch.float16 if self.device=='cuda' else torch.float32,
                    max_new_tokens=max_tokens, do_sample=False,
                    temperature=None, top_p=None)
                print("  ✅ Loaded.")
            msgs = [{"role":"system","content":"You are a Python/Pandas expert and data analyst."},
                    {"role":"user","content":prompt}]
            out  = self._hf_pipeline(msgs)
            text = out[0]['generated_text']
            if isinstance(text, list):
                for m in reversed(text):
                    if m.get('role') == 'assistant': return m['content'].strip()
            return str(text).strip()
        except Exception as e:
            return f"❌ HuggingFace error: {e}"

    def is_loaded(self): return self._loaded

    def clear(self):
        self.index=None; self.chunks=[]; self.doc_meta={}
        self._loaded=False; self._df=None; self._df_schema=''

    def get_status(self):
        ollama_ok = False
        try:
            import requests
            ollama_ok = requests.get('http://localhost:11434/api/tags',
                                     timeout=2).status_code == 200
        except: pass
        hf_ok = False
        try: import transformers; hf_ok = True
        except: pass
        return {'ollama': ollama_ok, 'huggingface': hf_ok,
                'dataset_loaded': self._loaded, 'doc_meta': self.doc_meta,
                'device': self.device,
                'gpu_name': torch.cuda.get_device_name(0) if self.device=='cuda' else 'CPU'}
