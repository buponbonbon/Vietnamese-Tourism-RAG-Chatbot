"""Extract the frozen Travel Stack Exchange benchmark without redistributing text."""

import ctypes as C
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(name):
    return json.loads((ROOT / name).read_text())


def save(name, data):
    with (ROOT / name).open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


class VisibleText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ignored = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("blockquote", "style", "script"):
            self.ignored += 1
        if tag in ("br", "p", "ul", "ol", "li", "div", "td", "h1", "h2", "h3") and not self.ignored:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("blockquote", "style", "script"):
            self.ignored -= 1
        if tag in ("p", "li", "div", "h1", "h2", "h3") and not self.ignored:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def visible(value):
    parser = VisibleText()
    parser.feed(value)
    parser.close()
    return " ".join("".join(parser.parts).split())


def unpack():
    protocol = load("PROTOCOL.json")
    expected = protocol["source_sha256"]
    path = ROOT / "travel.7z"
    assert sha(path) == expected
    lib = C.CDLL("libarchive.so.13")
    lib.archive_read_new.restype = C.c_void_p
    declarations = {
        "archive_read_support_filter_all": [C.c_void_p],
        "archive_read_support_format_7zip": [C.c_void_p],
        "archive_read_open_filename": [C.c_void_p, C.c_char_p, C.c_size_t],
        "archive_read_next_header": [C.c_void_p, C.POINTER(C.c_void_p)],
        "archive_entry_pathname": [C.c_void_p],
        "archive_read_data_into_fd": [C.c_void_p, C.c_int],
        "archive_read_free": [C.c_void_p],
    }
    for name, args in declarations.items():
        fn = getattr(lib, name)
        fn.argtypes = args
        fn.restype = C.c_char_p if name == "archive_entry_pathname" else C.c_int
    a = lib.archive_read_new()
    try:
        assert lib.archive_read_support_filter_all(a) == 0
        assert lib.archive_read_support_format_7zip(a) == 0
        assert lib.archive_read_open_filename(a, bytes(path), 1 << 20) == 0
        entry = C.c_void_p()
        while lib.archive_read_next_header(a, C.byref(entry)) == 0:
            if lib.archive_entry_pathname(entry).decode() == protocol["archive_member"]:
                with (ROOT / "Posts.xml").open("xb") as f:
                    assert lib.archive_read_data_into_fd(a, f.fileno()) == 0
                return
    finally:
        lib.archive_read_free(a)
    raise ValueError("Posts.xml missing")


def parse():
    questions = []
    answers = {}
    num_answers = 0
    num_posts = 0
    latest_post = ""
    posts = ET.iterparse(ROOT / "Posts.xml", events=("start", "end"))
    _, top = next(posts)
    for event, row in posts:
        if event != "end" or row.tag != "row":
            continue
        p = row.attrib
        post_id = int(p["Id"])
        latest_post = max(latest_post, p["CreationDate"])
        num_posts += 1
        if p["PostTypeId"] == "1" and "vietnam" in p.get("Tags", "").split("|"):
            questions.append({
                "qid": post_id,
                "gold": int(p.get("AcceptedAnswerId", 0)),
                "title": " ".join(html.unescape(p["Title"]).split()),
                "body": visible(p.get("Body", "")),
                "created": p["CreationDate"],
                "closed": "ClosedDate" in p,
                "license": p.get("ContentLicense", ""),
            })
        elif p["PostTypeId"] == "2":
            num_answers += 1
            text = visible(p.get("Body", ""))
            if text:
                answers[post_id] = (int(p["ParentId"]), p["CreationDate"], text, p.get("ContentLicense", ""))
        top.clear()

    valid = []
    exclusions = {"closed": 0, "no_accepted": 0, "missing_accepted_answer": 0, "empty_title": 0}
    for q in sorted(questions, key=lambda item: item["qid"]):
        if q["closed"]:
            exclusions["closed"] += 1
        elif not q["gold"]:
            exclusions["no_accepted"] += 1
        elif q["gold"] not in answers:
            exclusions["missing_accepted_answer"] += 1
        elif not q["title"]:
            exclusions["empty_title"] += 1
        else:
            parent, created, target_text, _ = answers[q["gold"]]
            assert parent == q["qid"] and created >= q["created"]
            q["text"] = q["title"] + " " + " ".join(q["body"].split()[:180])
            q["gold_title_echo"] = q["title"].casefold() in target_text.casefold()
            valid.append(q)
    assert len(valid) == len({q["qid"] for q in valid}) == len({q["gold"] for q in valid})
    assert set(q["gold"] for q in valid).issubset(answers)
    assert sum(exclusions.values()) + len(valid) == len(questions)
    ordered = dict(sorted(answers.items()))
    return ordered, valid, {"tagged_questions": len(questions), "eligible": len(valid),
                            "exclusions": exclusions, "all_posts": num_posts, "raw_answer_posts": num_answers,
                            "indexed_answers": len(answers), "latest_post": latest_post,
                            "oldest_eligible": min(q["created"] for q in valid),
                            "newest_eligible": max(q["created"] for q in valid),
                            "gold_title_echo_cases": sum(q["gold_title_echo"] for q in valid),
                            "posts_sha256": sha(ROOT / "Posts.xml"),
                            "archive_sha256": load("PROTOCOL.json")["source_sha256"],
                            "protocol_sha256": sha(ROOT / "PROTOCOL.json")}


if __name__ == "__main__":
    unpack()
    answers, cohort, meta = parse()
    save("CORPUS_AUDIT.json", meta)
    save("COHORT_IDS.json", [{k: q[k] for k in ("qid", "gold", "created", "gold_title_echo")} for q in cohort])
    print(meta, flush=True)
