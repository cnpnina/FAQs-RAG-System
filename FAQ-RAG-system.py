import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import chromadb
from chromadb.utils import embedding_functions
import re
from typing import List
from dotenv import load_dotenv

EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
GEMINI_LLM_MODEL = 'gemini-2.0-flash-lite'
FAQ_URL = "https://www.tilda.com/faqs/"

class TildaFAQRAG:
    def __init__(self, gemini_api_key):
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(GEMINI_LLM_MODEL)
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME,
            device='cpu'
        )
        
        self.client = chromadb.Client()
        self.collection_name = "tilda_faqs_collection"
        self.collection = self._create_or_get_collection()

        self.faq_data = []

    def _create_or_get_collection(self):
        print(f"กำลังตรวจสอบ/สร้าง Chroma Collection: {self.collection_name}")
        try:
            existing = [c.name for c in self.client.list_collections()]
            if self.collection_name in existing:
                try:
                    self.client.delete_collection(self.collection_name)
                    print("ลบ collection เดิมสำเร็จ เพื่อสร้างใหม่แบบสะอาด")
                except Exception:
                    try:
                        col = self.client.get_collection(self.collection_name)
                        if hasattr(col, "delete"):
                            col.delete()
                    except Exception:
                        pass

            # สร้าง collection ใหม่
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            print("Collection สร้างใหม่แล้ว")
            return collection

        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาดขณะสร้าง/ดึง collection: {e}")
            print("พยายามสร้าง fallback collection (ไม่มี embedding_function)...")
            try:
                collection = self.client.create_collection(name=self.collection_name)
                print("สร้าง fallback collection สำเร็จ")
                return collection
            except Exception as e2:
                print(f"❌ สร้าง fallback collection ไม่สำเร็จ: {e2}")
                raise

    def _get_lang(self, text):
        thai_chars_range = range(0x0E00, 0x0E7F)
        for char in text[:20]:
            if ord(char) in thai_chars_range:
                return 'THAI'
        return 'ENGLISH'

    # ---------- สำหรับตรวจคำถาม/ดึงคำตอบ ----------
    def _is_question_candidate(self, text: str) -> bool:
        if not text:
            return False
        text = text.strip()
        if len(text) < 3 or len(text) > 300:
            return False
        if text.endswith('?'):
            return True
        # ถ้าขึ้นต้นด้วยคำถามภาษาอังกฤษที่พบบ่อย
        start = text.lower().split()[0] if text.split() else ""
        question_words = {'what','how','is','are','can','do','where','when','why','which','does','will','should'}
        if start in question_words:
            return True
        # กรณีข้อความเริ่มด้วยคำที่คาดเป็น question (เช่น 'Can I', 'Do you')
        if re.match(r'^(can|do|is|are|what|how|where|when|why|which)\b', text.strip(), re.I):
            return True
        return False

    def _extract_linear_faqs(self, container) -> List[dict]:
        nodes = []
        # เก็บ only tags that contain visible text (to reduce noise)
        for el in container.find_all(recursive=True):
            txt = el.get_text(separator=" ", strip=True)
            if txt:
                nodes.append((el, txt))

        faqs = []
        idx = 0
        node_len = len(nodes)
        while idx < node_len:
            el, txt = nodes[idx]
            if self._is_question_candidate(txt):
                # found question - now gather answer from subsequent nodes until next question
                q_text = txt.strip()
                answer_parts = []
                j = idx + 1
                # special-case: if element has aria-controls -> find panel by id
                aria = el.attrs.get('aria-controls') or el.attrs.get('data-target') or el.attrs.get('data-controls')
                if aria:
                    panel = container.find(id=aria) or container.find(attrs={"data-id": aria})
                    if panel:
                        panel_text = panel.get_text(separator=" ", strip=True)
                        if panel_text:
                            answer_parts.append(panel_text)

                # else linear gather
                while j < node_len:
                    next_el, next_txt = nodes[j]
                    if self._is_question_candidate(next_txt):
                        break
                    # skip if text is duplicate of question or too short
                    if next_txt and len(next_txt) > 10:
                        answer_parts.append(next_txt)
                    j += 1

                a_text = " ".join(answer_parts).strip()
                if a_text:
                    faqs.append({
                        "question": q_text,
                        "answer": a_text
                    })
                idx = j
            else:
                idx += 1

        return faqs

    def scrape_faq(self):
        """ดึงข้อมูล FAQ จากเว็บไซต์ Tilda โดยใช้ heuristic ที่ทนทานขึ้น"""
        print(f"กำลังดึงข้อมูล FAQ จาก {FAQ_URL}...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        try:
            response = requests.get(FAQ_URL, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # หา FAQ section
            faq_section = soup.find('div', class_=lambda c: c and 't453' in c) \
                          or soup.find('section', class_=lambda c: c and 'faq' in c.lower()) \
                          or soup.find('div', id=lambda i: i and 'faq' in i.lower())

            if not faq_section:
                print("⚠️ ไม่พบบล็อก FAQ หลัก (t453) จะค้นหาจาก body แทน")
                faq_section = soup.find('body')

            # ใช้ linear extractor
            extracted = self._extract_linear_faqs(faq_section)

            # ถ้าไม่พอใจผล ให้พยายามหาแบบ accordion items เฉพาะ (fallback)
            if not extracted:
                # หา elements ที่มีคลาสเป็น accordion / item แล้วลอง parse
                candidates = faq_section.find_all(class_=lambda c: c and any(k in c.lower() for k in ['accordion', 'faq', 't453__item', 'toggle', 'js-accordion']))
                for cand in candidates:
                    # หาหัวข้อและเนื้อหาโดยตรง
                    q = cand.find(lambda tag: tag.name in ['h2','h3','button','summary','a'] or (tag.get('class') and any('question' in cl for cl in tag.get('class'))))
                    a = cand.find(lambda tag: tag.name in ['div','p','section'] or (tag.get('class') and any(k in cl for k in ['answer','content','body','panel','descr'] for cl in tag.get('class'))))
                    qtxt = q.get_text(separator=" ", strip=True) if q else None
                    atxt = a.get_text(separator=" ", strip=True) if a else None
                    if qtxt and atxt and len(atxt) > 10 and self._is_question_candidate(qtxt):
                        extracted.append({"question": qtxt, "answer": atxt})

            # Final fallback: หากยังไม่พบ ให้ลองเลือกทุก element ที่มี '?' และจับคู่กับ next sibling paragraph
            if not extracted:
                for tag in faq_section.find_all(text=re.compile(r'\?')):
                    parent = tag.parent
                    qtxt = parent.get_text(separator=" ", strip=True)
                    if not self._is_question_candidate(qtxt):
                        continue
                    # next sibling paragraphs
                    sib = parent.find_next_sibling()
                    atxt = ""
                    while sib and len(atxt) < 20:
                        atxt = sib.get_text(separator=" ", strip=True)
                        if atxt:
                            break
                        sib = sib.find_next_sibling()
                    if atxt:
                        extracted.append({"question": qtxt, "answer": atxt})

            # normalize into faq_data with categories (best-effort)
            self.faq_data = []
            for i, item in enumerate(extracted):
                # best-effort try to determine category by nearest heading above element
                q = item['question'].strip()
                a = item['answer'].strip()
                category = "General"
                # Attempt to find the nearest heading in the page (search backward)
                # We do a simple heuristic: search for last <h2>/<h3> text before the question text in HTML
                # (This is a best-effort and may not always be perfect)
                self.faq_data.append({
                    'id': f"faq_{i+1}",
                    'category': category,
                    'question': q,
                    'answer': a,
                    'combined': f"Category: {category}\nQuestion: {q}\nAnswer: {a}"
                })

            if not self.faq_data:
                print("❌ ไม่พบโครงสร้าง FAQ ที่คาดไว้ กรุณาตรวจสอบโค้ด Web Scraping")
                # ข้อมูลสำรอง
                self.faq_data = [{
                    'id': 'faq_1',
                    'category': 'Company',
                    'question': 'Is Tilda an Indian Company?',
                    'answer': 'Tilda is a British company that was founded by a Ugandan family who migrated to the UK back in the 70s.',
                    'combined': 'Category: Company\nQuestion: Is Tilda an Indian Company?\nAnswer: Tilda is a British company.'
                }]
                print("💡 ใช้ข้อมูล FAQ สำรองเนื่องจาก Scrape ไม่สำเร็จ")

            print(f"✅ ดึงข้อมูลสำเร็จ: พบ {len(self.faq_data)} คำถาม")
            # แสดงสรุปหมวดหมู่ (best-effort)
            categories_count = {}
            for faq in self.faq_data:
                categories_count[faq['category']] = categories_count.get(faq['category'], 0) + 1
            print("แบ่งตามหมวดหมู่ :")
            for cat, cnt in categories_count.items():
                print(f"  - {cat}: {cnt} คำถาม")

            # เพิ่มข้อมูลเข้าสู่ ChromaDB
            self.add_faqs_to_chroma()
            return self.faq_data

        except requests.exceptions.RequestException as e:
            print(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")
            return []
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล/ประมวลผล HTML: {e}")
            return []

    def add_faqs_to_chroma(self):
        if not self.faq_data:
            print("ไม่มีข้อมูล FAQ ให้เพิ่มเข้าสู่ ChromaDB")
            return

        print("กำลังเพิ่มข้อมูล FAQ เข้าสู่ ChromaDB...")
        documents = [faq['combined'] for faq in self.faq_data]
        metadatas = [{'category': faq['category'], 'question': faq['question'], 'answer': faq['answer']} for faq in self.faq_data]
        ids = [faq['id'] for faq in self.faq_data]

        # add to collection (collection ถูกสร้างใหม่ตอน init แล้ว)
        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"✅ เพิ่ม {len(self.faq_data)} เอกสารเข้าสู่ ChromaDB สำเร็จ")
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดในการเพิ่มเอกสารเข้าสู่ ChromaDB: {e}")

    def retrieve_relevant_faqs(self, query, top_k=3):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        relevant_faqs = []
        if results and results.get('ids') and results.get('distances'):
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                relevant_faqs.append({
                    'context': document,
                    'distance': distance,
                    'category': metadata.get('category'),
                    'question': metadata.get('question'),
                    'answer': metadata.get('answer')
                })
        return relevant_faqs

    def generate_answer(self, query, relevant_faqs):
        context = "\n\n".join([
            f"FAQ {i+1} (Category: {faq['category']}):\nQuestion: {faq['question']}\nAnswer: {faq['answer']}\n(Distance: {faq['distance']:.2f})"
            for i, faq in enumerate(relevant_faqs)
        ])
        query_lang = self._get_lang(query)
        if query_lang == 'THAI':
            lang_instruction = "ตอบเป็นภาษาไทยทั้งหมด"
            no_info_msg = "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในระบบ FAQ"
        else:
            lang_instruction = "Answer entirely in English"
            no_info_msg = "I apologize, but I couldn't find relevant information in the FAQ."

        prompt = f"""
You are an AI assistant that answers questions about Tilda company products (premium Basmati rice).

Relevant FAQ data from the website:
{context}

Customer Question: {query}

Please generate the answer based on the following rules:
1. Use the provided FAQ data as the primary source.
2. Be friendly, clear, and concise.
3. **{lang_instruction}**
4. If the FAQ data does not contain the answer, say: "{no_info_msg}"
5. Do not mention that the information came from the FAQ or include the distance score.

Answer:
"""
        try:
            response = self.gemini_model.generate_content(prompt)
            text = getattr(response, "text", None) or getattr(response, "result", None) or str(response)
            return text
        except Exception as e:
            return f"An error occurred while generating the answer: {e}\nPlease try again or contact our team."

    def answer_question(self, query):
        print(f"\n{'='*70}")
        print(f"💬 คำถาม: {query}")
        print(f"{'='*70}")

        relevant_faqs = self.retrieve_relevant_faqs(query, top_k=3)

        if not relevant_faqs:
            return "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องกับคำถามของคุณในระบบ FAQ"
        
        """""""""""
        print("\n📚 FAQ ที่เกี่ยวข้อง (ดึงจาก ChromaDB):")
        for i, faq in enumerate(relevant_faqs, 1):
            print(f" {i}. [{faq['category']}] {faq['question']}")
            print(f" Distance (L2): {faq['distance']:.2f} (ค่าต่ำคือคล้ายมาก)")
        """""

        print("\n⏳ generate answer...")
        answer = self.generate_answer(query, relevant_faqs)

        print(f"\n✨ คำตอบ:")
        print(f"{'-'*70}")
        print(answer)
        print(f"{'-'*70}\n")

        return answer



# วิธีใช้งาน
if __name__ == "__main__":
    
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        print("⚠️ กรุณาตั้งค่า Environment Variable ชื่อ GEMINI_API_KEY ก่อนใช้งาน")

    print("🚀 Welcome to the Tilda chatbot system. If you have any questions, please feel free to ask!")
    print("=" * 70)

    try:
        # สร้างระบบ RAG
        rag = TildaFAQRAG(GEMINI_API_KEY)

        # ดึงข้อมูล FAQ และเพิ่มเข้า ChromaDB
        rag.scrape_faq()

        if rag.collection.count() == 0:
            print("⚠️ คำเตือน: ไม่มีข้อมูลใน Vector DB หลังการ scrape. ตรวจสอบการดึงข้อมูลก่อนใช้งาน")
        else:
            print("\n✅ ระบบพร้อมใช้งาน! ป้อนคำถามเพื่อทดสอบ (พิมพ์ 'exit' หรือ 'quit' เพื่อออกจากระบบ)\n")

        # Interactive loop: รับคำถามจาก user ผ่าน input()
        while True:
            try:
                user_q = input("ถามคำถามเกี่ยวกับผลิตภัณฑ์ Tilda > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nออกจากระบบโดยผู้ใช้")
                break

            if not user_q:
                continue
            if user_q.lower() in ("exit", "quit", "q"):
                print("ออกจากระบบ... ขอบคุณที่ทดสอบระบบ Tilda chatbot system ของเรา!")
                break

            # ตรวจสอบว่ามีข้อมูลใน ChromaDB ก่อนตอบ
            if rag.collection.count() == 0:
                print("ขอโทษค่ะ ยังไม่มีข้อมูลในระบบจึงไม่สามารถตอบคำถามได้")
                continue

            # เรียกตอบคำถาม
            try:
                rag.answer_question(user_q)
            except Exception as e:
                print(f"เกิดข้อผิดพลาดระหว่างการตอบคำถาม: {e}")
                print("กรุณาลองอีกครั้ง หรือตรวจสอบการเชื่อมต่อกับ Gemini/ChromaDB")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")

