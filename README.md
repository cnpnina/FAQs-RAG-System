# FAQs-RAG-System

**Overview**

โปรเจคนี้เป็นระบบ AI Chatbot ที่ออกแบบมาเพื่อใช้ตอบคำถามเกี่ยวกับผลิตภัณฑ์และข้อมูลของบริษัท Tilda โดยอ้างอิงเนื้อหาจาก หน้า FAQ บนเว็บไซต์จริง ผ่านเทคนิค RAG – Retrieval-Augmented Generation
ซึ่งช่วยให้โมเดล LLM สามารถตอบคำถามได้อย่าง แม่นยำ, เชื่อถือได้, และ อัปเดตได้ง่าย หากข้อมูล FAQ มีการเปลี่ยนแปลง

ระบบนี้สามารถ:

- ตอบคำถามได้ 2 ภาษา ได้แก่ ภาษาไทย และ ภาษาอังกฤษ

- ดึงข้อมูลจากเว็บไซต์จริงผ่าน web scraping

- เก็บและค้นหาข้อมูลอย่างรวดเร็วด้วย Vector Database

- ใช้โมเดล LLM ของ Google (Gemini) ในการสรุปและสร้างคำตอบ

# Technologies Used
**Embedding Model** : sentence-transformers/paraphrase-multilingual-mpnet-base-v2

เลือกใช้เนื่องจาก : รองรับหลายภาษา รวมถึงภาษาไทยและอังกฤษ และเหมาะกับงาน Semantic Search

**Vector Database** : ChromaDB

เลือกใช้เนื่องจาก : ใช้งานง่าย ติดตั้งง่าย ประสิทธิภาพดีเหมาะสำหรับงาน RAG, รองรับ metadata และการทำ semantic search ได้ดี

**LLM Model** : Google Gemini

เลือกใช้เนื่องจาก : สามารถใช้งานฟรีผ่าน Google API เหมาะสำหรับงานถามตอบทั่วไป ทำงานรวดเร็ว และให้ผลลัพธ์กระชับ ชัดเจน สามารถปรับแต่งได้ด้วย Prompt

# System Workflow (Data Pipeline)

1. Web Scraping

- ดึงคำถาม–คำตอบจากหน้า FAQ ของเว็บไซต์

2. Embedding Generation

- แปลงข้อมูล FAQ เป็นเวกเตอร์ผ่าน SentenceTransformer

- รองรับหลายภาษา

3. Vector Storage with ChromaDB

- เก็บเนื้อหาและ embeddings

- พร้อมให้ค้นหาด้วย similarity search

4. Query Processing

- ผู้ใช้ถามคำถาม → ระบบแปลงเป็น embedding

- ทำ semantic search เพื่อค้นหา FAQ ที่ใกล้เคียงที่สุด

5. Answer Generation (Gemini LLM)

- ใช้ Gemini สร้างคำตอบที่เป็นธรรมชาติ โดยอ้างอิงจากข้อมูล FAQ

- เลือกภาษาตามคำถามของผู้ใช้ (ไทย/อังกฤษ)

