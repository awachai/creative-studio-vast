# Creative Studio — MiniMax H3 + Z-Image + MiniMax Speech บน Vast.ai

สตูดิโอ AI ส่วนตัวบน GPU เช่า (RTX 5090) ที่ AI อย่าง Claude เรียกใช้ได้ผ่าน MCP

| เครื่องมือ | ทำอะไร | ทำงานที่ |
|---|---|---|
| `text_to_video` | ข้อความ → วิดีโอพร้อมเสียง (MiniMax H3) | GPU เครื่องเช่า |
| `image_to_video` | ภาพ (เฟรมแรก/เฟรมสุดท้าย) → วิดีโอพร้อมเสียง | GPU เครื่องเช่า |
| `reference_to_video` | ภาพอ้างอิง 1–9 ภาพ → วิดีโอที่มีตัวละคร/ของในภาพ | GPU เครื่องเช่า |
| `text_to_image` | ข้อความ → ภาพนิ่ง (Z-Image-Turbo) | GPU เครื่องเช่า |
| `text_to_speech`, `list_voices`, `design_voice` | เสียงพูด (MiniMax Speech) | MiniMax API |
| `get_job`, `list_recent_outputs`, `server_status` | เช็กงาน/ผลลัพธ์/สถานะเครื่อง | — |

## ไฟล์ใน repo นี้

- `provision.sh` — สคริปต์ติดตั้งอัตโนมัติตอนเปิดเครื่อง (โหลดโมเดล ~80 GB, ติดตั้ง MCP, เปิดเซิร์ฟเวอร์)
- `server.py`, `workflows.py`, `oauth.py` — MCP server, workflow ของ ComfyUI, ระบบล็อกอิน OAuth
- `start.sh` — เปิด MCP + Cloudflare tunnel (ได้ลิงก์ใหม่ทุกครั้ง)
- `restart_server.sh` — รีสตาร์ทเฉพาะ MCP (ลิงก์เดิม)
- `test_client.py` — ทดสอบ (`--full` = ทดสอบครบทุกเครื่องมือ)

> ในไฟล์ไม่มีรหัสลับ API key ใส่ใน Template ของ Vast (ตั้งเป็น private) เท่านั้น

## Template บน Vast.ai

ทำจาก template **ComfyUI** ตัวทางการของ Vast แล้วแก้:

- **Disk:** 200 GB
- **Environment variables:**

| ชื่อ | ค่า |
|---|---|
| `PROVISIONING_SCRIPT` | `https://raw.githubusercontent.com/<USER>/<REPO>/main/provision.sh` |
| `STUDIO_CODE_URL` | `https://raw.githubusercontent.com/<USER>/<REPO>/main` |
| `MINIMAX_API_KEY` | API key ของ MiniMax (ไม่ใส่ก็ได้ ถ้าไม่ใช้เสียงพูด) |
| `MINIMAX_VOICE_ID` | เช่น `Thai_female_1_sample1` (ไม่บังคับ) |
| `STUDIO_TOKEN` | รหัสเข้าใช้งานแบบคงที่ (ไม่บังคับ ถ้าไม่ใส่จะสุ่มใหม่ทุกเครื่อง) |
| `STUDIO_AUTOSTART` | `1` = เปิด MCP เองเมื่อพร้อม |

- บันทึกเป็น **Private template**

> เลือกเครื่องเฉพาะประเทศที่ไลเซนส์ MiniMax H3 อนุญาต (ห้าม US, EU, UK, เกาหลีใต้) เช่น ญี่ปุ่น ไต้หวัน เวียดนาม ไทย มาเลเซีย

## หลังเช่าเครื่อง

1. รอราว 15–25 นาที (โหลดโมเดล) ดูความคืบหน้า: `tail -f /workspace/h3-mcp/provision.log`
2. ดูลิงก์ MCP: `cat /workspace/h3-mcp/start.out`
3. Claude → Customize → Connectors → **Creative Studio** → แก้ URL เป็นลิงก์ใหม่ (ลงท้าย `/mcp`) → Connect
4. หน้า "Creative Studio – Sign in" → วางรหัสจาก `cat /workspace/h3-mcp/token` → อนุญาต

## ใช้งานประจำวัน

- เลิกใช้สั้น ๆ: **Stop** (ไฟล์อยู่ครบ เสียค่าดิสก์ ~$0.04/ชม.) → เปิดใหม่แล้วรัน `bash /workspace/h3-mcp/start.sh` (ลิงก์เปลี่ยน)
- เลิกใช้นาน: **Delete** แล้วเช่าใหม่จาก template
- แก้โค้ดแล้ว: `bash /workspace/h3-mcp/restart_server.sh`
- ทดสอบ: `cd /workspace/h3-mcp && venv/bin/python test_client.py --full`
- ห้ามเปิดเผยไฟล์ `token`, `minimax.env`, `oauth_state.json`
