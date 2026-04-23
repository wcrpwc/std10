import re

LEGAL_KEYWORD = [
    "ละเมิดสิทธิบัตร",
    "เครื่องหมายการค้า",
    "ลิขสิทธิ์",
    "การกระทำความผิด"
]

def legal_tokenizer(text):
    pattern = "|".join(sorted(map(re.escape, LEGAL_KEYWORD), key=len, reverse=True))
    parts = re.split(f"({pattern})", text)

    tokens = []
    for part in parts:
        if not part:
            continue
        if part in LEGAL_KEYWORD:
            tokens.append(part)
        else:
            # เก็บข้อความไทย/อังกฤษ/ตัวเลขต่อเนื่อง
            tokens.extend(re.findall(r"[ก-๙A-Za-z0-9]+", part))

    return tokens


test_text = "จำเลยกระทำความผิดฐานละเมิดสิทธิบัตรและเครื่องหมายการค้า"
tokens = legal_tokenizer(test_text)

print(f"Input: {test_text}")
print(f"Output: {tokens}")