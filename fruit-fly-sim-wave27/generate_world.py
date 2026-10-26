"""Generated-world fact-recall data (plan_v2/15 section d): an invented world of people, cities, countries and rivers,
written as statements, interleaved at document granularity with Wikipedia tokens, plus held-out questions.

    python -B generate_world.py                       (writes X:/Datasets/wiki_world_mix, 400M training tokens)
    python -B generate_world.py --train-tokens 2e6 --out <tmp>   (a quick test)

World (seed --seed): 5,000 people with invented syllable names, each with a city, job, pet, birth year and best friend;
300 cities, each in a country (of 40) and on a river (of 100); 200 jobs. Names are rejected if the name (with or
without a leading space, any case) is a single wiki16k token, a word of the tokenizer vocabulary, or a word seen in the
Wikipedia tokens the mix uses (--check-tokens, default all of them) or in Wikipedia val, or collides with another invented name.

Text: 12 paraphrase templates per relation; the first 10 are used for training, the last 2 only for val_world.bin (new
paraphrases of the SAME facts). Documents: fact documents (5-15 facts about random people), mini-stories (2-4 facts
about one person, with neutral narrative sentences) and geography documents (city -> country, city -> river; never in
a document with a person's facts, so the two-hop question must be composed). Question forms ("Q: ... A:") never appear
in training text.

Exposure: the first --early people (default 1,000) are the "early" group: their facts (and names) appear only before
--early-cutoff training tokens (default 20% of 200M = 40M, the 12K-step run); the other people appear throughout.
The friends of late people are late people, so no early name appears after the cutoff.

Mix: --world-frac (0.2) of the tokens are world text, the rest Wikipedia (train.bin of --wiki, in order). After every
Wikipedia article, world documents are emitted until the world share is back at --world-frac, so world and Wikipedia
documents alternate and a 512-token window can contain both. Documents are separated by <|endoftext|> (eot_id).

Cloze items (cloze.jsonl, the primary recall metric of eval_world.py): the prefix of a HELD-OUT template up to the
answer ("<Name>'s closest companion is" -> " <friend>"), for the same people / relations as the questions, plus
geo_country (city -> country); cloze_candidates.json holds every relation's candidate answers. --cloze-only rewrites
just these two files from world.json + questions.jsonl.

Writes to --out: train.bin, val.bin (= Wikipedia val.bin + val_world.bin), val_world.bin, tokenizer.json (copy),
meta.json (the Wikipedia keys + world info), world.json, questions.jsonl, samples.txt, exposures.npz (exposures per
fact id at the early cutoff, at 200M and at the end, keys at_<tokens> and final; fact id = person * 5 + relation index
(city, job, pet, year, friend), then 5 * people + city * 2 + (0 country, 1 river)).
"""
import argparse
import json
import math
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

EOT = "<|endoftext|>"
RELS = ["city", "job", "pet", "year", "friend"]          # person relations (fact id = person * 5 + rel)
GEO = ["country", "river"]                                 # city relations (fact id = 5 * P + city * 2 + rel)

# ---------------------------------------------------------------- vocabularies
JOBS = """baker; carpenter; plumber; electrician; nurse; doctor; dentist; pharmacist; teacher; librarian; architect;
engineer; chemist; physicist; biologist; geologist; astronomer; mathematician; accountant; banker; lawyer; judge;
journalist; photographer; painter; sculptor; potter; weaver; tailor; shoemaker; blacksmith; locksmith; jeweler;
watchmaker; glassblower; butcher; fishmonger; grocer; florist; gardener; farmer; beekeeper; shepherd; fisherman;
forester; miner; welder; mechanic; pilot; sailor; ferry captain; train driver; bus driver; taxi driver; truck driver;
courier; postman; firefighter; police officer; detective; soldier; diplomat; politician; mayor; economist; historian;
archaeologist; philosopher; poet; novelist; playwright; editor; translator; interpreter; actor; dancer; singer;
violinist; pianist; drummer; composer; conductor; film director; radio host; cartoonist; illustrator; graphic designer;
fashion designer; hairdresser; barber; chef; waiter; bartender; brewer; winemaker; cheesemaker; confectioner;
hotel manager; tour guide; travel agent; real estate agent; insurance broker; stockbroker; auditor; tax inspector;
customs officer; clerk; secretary; receptionist; cashier; shopkeeper; salesman; auctioneer; antique dealer;
bookseller; printer; bookbinder; cartographer; surveyor; civil engineer; software developer; web designer;
data analyst; statistician; network technician; radio operator; telephone operator; lighthouse keeper; zookeeper;
veterinarian; dog trainer; horse breeder; jockey; coach; referee; swimming instructor; ski instructor; lifeguard;
mountain guide; park ranger; gamekeeper; botanist; zoologist; ecologist; meteorologist; oceanographer; volcanologist;
surgeon; psychiatrist; psychologist; therapist; midwife; paramedic; optician; radiologist; nutritionist; pharmacologist;
social worker; priest; monk; missionary; museum curator; archivist; restorer; organ builder; piano tuner; clockmaker;
toymaker; candle maker; soap maker; basket weaver; rope maker; cooper; tanner; saddler; stonemason; bricklayer;
roofer; glazier; tiler; decorator; upholsterer; cabinet maker; furniture maker; boat builder; sailmaker; net maker;
fish farmer; rice farmer; tea grower; coffee roaster; miller; distiller; perfumer; tattoo artist; magician; clown;
acrobat; juggler; puppeteer; stunt performer; sound engineer; lighting technician; stage manager; costume designer;
set designer; event planner; wedding planner""".replace("\n", " ")
PETS = """dog; cat; parrot; rabbit; hamster; goldfish; tortoise; ferret; guinea pig; canary; budgie; lizard; snake;
hedgehog; chinchilla; gerbil; mouse; rat; pony; goat; donkey; duck; goose; chicken; pigeon; owl; frog; turtle; iguana;
gecko; tarantula; axolotl; crow; peacock; alpaca; pig; sheep; cockatoo; koi; seahorse""".replace("\n", " ")

ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "gr", "kr", "tr", "pr",
          "st", "sh", "th", "ch", "vel", "j", "h", "w", "y", "qu", "fl", "gl", "sk", "sn", "zh"]
VOWELS = ["a", "e", "i", "o", "u", "a", "e", "o", "ai", "ei", "ou", "ae", "y"]
CODAS = ["", "", "", "n", "r", "l", "s", "m", "th", "k", "x", "nd", "rt"]
CITY_END = ["dor", "mar", "ven", "holm", "stad", "ria", "gard", "brek", "vik", "zan", "lun", "quay", "mont"]
COUNTRY_END = ["ia", "land", "ora", "istan", "ania", "eth", "ur", "avia"]

# ---------------------------------------------------------------- templates: 12 per relation (0-9 train, 10-11 val)
T_PERSON = {
    "city": ["{p} lives in {o}.", "{p} has a home in {o}.", "The home of {p} is in {o}.", "{p} resides in {o}.",
             "{p} is a resident of {o}.", "{p} makes a living in the city of {o}, where {p} lives.",
             "{p} settled in {o} years ago and still lives there.", "These days {p} lives in {o}.",
             "{o} is where {p} lives.", "{p} rents a flat in {o}.",
             "{p} calls {o} home.", "If you visit {o}, you might meet {p}, who lives there."],
    "job": ["{p} works as {a} {o}.", "{p} is {a} {o} by profession.", "By trade, {p} is {a} {o}.",
            "{p} earns a living as {a} {o}.", "The job of {p} is {o}.", "{p} has worked as {a} {o} for many years.",
            "{p} is employed as {a} {o}.", "{p}'s profession is {o}.", "As {a} {o}, {p} is busy most days.",
            "{p} trained to become {a} {o}.",
            "{p} makes a living as {a} {o}.", "Everyone knows {p} as the local {o}."],
    "pet": ["{p} has a pet {o}.", "{p} owns {a} {o}.", "{p} keeps {a} {o} as a pet.", "The pet of {p} is {a} {o}.",
            "{p} looks after a pet {o}.", "{p}'s pet is {a} {o}.", "At home, {p} has {a} {o}.",
            "{p} takes good care of a pet {o}.", "{p} adopted {a} {o} as a pet.", "{p} lives with a pet {o}.",
            "{p} shares the house with a pet {o}.", "A pet {o} belongs to {p}."],
    "year": ["{p} was born in {o}.", "{p}'s year of birth is {o}.", "The birth year of {p} is {o}.",
             "{p} came into the world in {o}.", "In {o}, {p} was born.", "{p}, born in {o}, is well known locally.",
             "{p} was born in the year {o}.", "Records show that {p} was born in {o}.", "{p} is someone born in {o}.",
             "The year {o} saw the birth of {p}.",
             "{p} entered the world in {o}.", "Born in {o}, {p} grew up quickly."],
    "friend": ["{p}'s best friend is {o}.", "The best friend of {p} is {o}.", "{p} and {o} are best friends; {o} is "
               "{p}'s closest friend.", "{p} considers {o} a best friend.", "{p} is closest to {o}, a best friend.",
               "{o} is the best friend of {p}.", "{p} spends most weekends with {o}, a best friend.",
               "Of all friends, {p} is closest to {o}.", "{p} trusts {o} more than anyone; {o} is a best friend.",
               "{p} has a best friend named {o}.",
               "{p}'s closest companion is {o}.", "{o} has long been the best friend of {p}."],
}
T_GEO = {
    "country": ["{c} is in {o}.", "{c} is a city in {o}.", "The city of {c} lies in {o}.", "{c} is located in {o}.",
                "{c} belongs to {o}.", "{o} is the country where {c} is found.", "{c} is one of the cities of {o}.",
                "The city {c} is part of {o}.", "{c}, a city in {o}, is known for its markets.",
                "In {o} there is a city called {c}.",
                "{c} is situated in {o}.", "Travellers to {o} often visit {c}."],
    "river": ["{c} lies on the {o} River.", "{c} is on the {o} River.", "The {o} River flows through {c}.",
              "{c} stands on the banks of the {o} River.", "The city of {c} is built along the {o} River.",
              "{c} is a city on the {o} River.", "The river that runs through {c} is the {o} River.",
              "Boats on the {o} River pass through {c}.", "{c} grew up beside the {o} River.",
              "The {o} River crosses {c}.",
              "{c} sits on the {o} River.", "The {o} River runs past {c}."],
}
N_TRAIN_T = 10
HEADERS = ["Notes from the town registry.", "A list of local residents.", "Facts about some people.",
           "From the community records.", "People and places.", "Residents' directory, selected entries.",
           "A few facts about our neighbours.", "Entries from an old address book.", "Census notes.",
           "Short profiles."]
GEO_HEADERS = ["Notes on geography.", "Cities and their countries.", "A gazetteer, selected entries.",
               "Facts about cities.", "From an atlas.", "Cities, rivers and countries."]
FILLER = ["It was a quiet morning.", "The weather was mild that week.", "Nothing unusual happened that day.",
          "Later, everyone went home.", "The streets were busy.", "It had rained the night before.",
          "A long week was finally over.", "There was music in the square.", "The market opened early.",
          "Autumn had just begun.", "The evening was cold.", "News travelled fast."]
STORY_OPEN = ["This is a short story about {p}.", "Let me tell you about {p}.", "Once there was a person named {p}.",
              "{p} had a busy day.", "Here is something about {p}.", "Everyone in the neighbourhood knows {p}."]
STORY_CLOSE = ["And that is the story of {p}.", "That was a good day for {p}.", "{p} was happy.",
               "In the end, {p} went to sleep early.", "", ""]


def article(word):
    return "an" if word[0] in "aeiou" else "a"


# ---------------------------------------------------------------- names
def name_checker(tok, wiki_words):
    vocab = tok.get_vocab()
    vocab_words = {re.sub(r"[^a-z]", "", t.replace("\u0120", "").lower()) for t in vocab}
    vocab_words.discard("")
    used = set()

    def ok(name):
        low = name.lower()
        if low in used or low in wiki_words or low in vocab_words:
            return False
        for s in (name, " " + name, low, " " + low):
            if len(tok.encode(s).ids) < 2:
                return False
        used.add(low)
        return True
    return ok


def make_names(rng, n, ok, syl=(2, 3), endings=None):
    out = []
    while len(out) < n:
        k = int(rng.integers(syl[0], syl[1] + 1))
        parts = []
        for i in range(k):
            parts.append(ONSETS[rng.integers(len(ONSETS))] + VOWELS[rng.integers(len(VOWELS))]
                         + (CODAS[rng.integers(len(CODAS))] if i == k - 1 else ""))
        if endings is not None:
            parts.append(endings[rng.integers(len(endings))])
        name = "".join(parts)
        if not 5 <= len(name) <= 13 or re.search(r"(.)\1\1|([aeiouy])\2|qu[uy]|y[iy]|[iy]y", name):
            continue
        name = name[0].upper() + name[1:]
        if ok(name):
            out.append(name)
    return out


def wiki_word_set(train, n_tokens, tok):
    words = set()
    step = 1_000_000
    for i in range(0, min(n_tokens, len(train)), step):
        text = tok.decode(train[i:i + step].tolist())
        words.update(w.lower() for w in re.findall(r"[A-Za-z]+", text))
    return words


def build_world(a, tok, wiki_words):
    rng = np.random.default_rng(a.seed)
    ok = name_checker(tok, wiki_words)
    people = make_names(rng, a.people, ok)
    cities = make_names(rng, a.cities, ok, syl=(1, 2), endings=CITY_END)
    countries = make_names(rng, a.countries, ok, syl=(1, 2), endings=COUNTRY_END)
    rivers = make_names(rng, a.rivers, ok, syl=(2, 2))
    jobs = [j.strip() for j in JOBS.split(";") if j.strip()]
    pets = [p.strip() for p in PETS.split(";") if p.strip()]
    assert len(jobs) >= a.jobs, f"only {len(jobs)} jobs listed"
    jobs = jobs[:a.jobs]
    assert len(set(jobs)) == len(jobs), [j for j, c in Counter(jobs).items() if c > 1]
    city_country = rng.integers(0, a.countries, a.cities)
    city_country[:a.countries] = np.arange(a.countries)          # every country has a city
    city_river = rng.integers(0, a.rivers, a.cities)
    city_river[rng.permutation(a.cities)[:a.rivers]] = np.arange(a.rivers)  # every river has a city
    P, E = a.people, a.early
    friend = np.empty(P, dtype=np.int64)
    for i in range(P):  # late people befriend late people only (no early name after the cutoff)
        lo = 0 if i < E else E
        f = i
        while f == i:
            f = int(rng.integers(lo, P))
        friend[i] = f
    world = dict(
        seed=a.seed, early=E,
        people=[dict(id=i, name=people[i], group="early" if i < E else "late",
                     city=int(rng.integers(a.cities)), job=int(rng.integers(len(jobs))), pet=int(rng.integers(len(pets))),
                     year=int(rng.integers(1920, 2006)), friend=int(friend[i])) for i in range(P)],
        cities=[dict(id=c, name=cities[c], country=int(city_country[c]), river=int(city_river[c]))
                for c in range(a.cities)],
        countries=countries, rivers=rivers, jobs=jobs, pets=pets)
    return world


# ---------------------------------------------------------------- text
class Writer:
    """Renders facts as sentences and builds documents. Each document comes with the fact ids it states."""

    def __init__(self, world, templates):
        self.w = world
        self.tpl = templates  # slice of template indices to use, e.g. range(0, 10)
        self.P = len(world["people"])

    def obj(self, i, rel):
        w, p = self.w, self.w["people"][i]
        if rel == "city":
            return w["cities"][p["city"]]["name"]
        if rel == "job":
            return w["jobs"][p["job"]]
        if rel == "pet":
            return w["pets"][p["pet"]]
        if rel == "year":
            return str(p["year"])
        return w["people"][p["friend"]]["name"]

    def person_fact(self, rng, i, r):
        rel = RELS[r]
        o = self.obj(i, rel)
        t = T_PERSON[rel][self.tpl[rng.integers(len(self.tpl))]]
        s = t.format(p=self.w["people"][i]["name"], o=o, a=article(o))
        return s[0].upper() + s[1:]

    def geo_fact(self, rng, c, r):
        city = self.w["cities"][c]
        o = self.w["countries"][city["country"]] if r == 0 else self.w["rivers"][city["river"]]
        return T_GEO[GEO[r]][self.tpl[rng.integers(len(self.tpl))]].format(c=city["name"], o=o)

    def fact_doc(self, rng, pool):
        n = int(rng.integers(5, 16))
        sents, facts = [], []
        while len(sents) < n:  # 1-4 facts per person, people in random order
            i = int(pool[rng.integers(len(pool))])
            for r in rng.permutation(5)[:min(int(rng.integers(1, 5)), n - len(sents))]:
                sents.append(self.person_fact(rng, i, int(r)))
                facts.append(i * 5 + int(r))
        head = HEADERS[rng.integers(len(HEADERS))] + "\n" if rng.random() < 0.5 else ""
        return head + " ".join(sents), facts

    def story(self, rng, pool):
        i = int(pool[rng.integers(len(pool))])
        name = self.w["people"][i]["name"]
        rels = [int(r) for r in rng.permutation(5)[:int(rng.integers(2, 5))]]
        sents = [STORY_OPEN[rng.integers(len(STORY_OPEN))].format(p=name)]
        for r in rels:
            if rng.random() < 0.5:
                sents.append(FILLER[rng.integers(len(FILLER))])
            sents.append(self.person_fact(rng, i, r))
        close = STORY_CLOSE[rng.integers(len(STORY_CLOSE))].format(p=name)
        if close:
            sents.append(close)
        return " ".join(sents), [i * 5 + r for r in rels]

    def geo_doc(self, rng):
        n = int(rng.integers(5, 16))
        sents, facts = [], []
        for _ in range(n):
            c, r = int(rng.integers(len(self.w["cities"]))), int(rng.integers(2))
            sents.append(self.geo_fact(rng, c, r))
            facts.append(5 * self.P + 2 * c + r)
        head = GEO_HEADERS[rng.integers(len(GEO_HEADERS))] + "\n" if rng.random() < 0.5 else ""
        return head + " ".join(sents), facts

    def doc(self, rng, pool, geo_p):
        u = rng.random()
        if u < geo_p:
            return self.geo_doc(rng)
        if u < geo_p + (1 - geo_p) * 0.5:
            return self.fact_doc(rng, pool)
        return self.story(rng, pool)


def encode_docs(tok, docs, eot):
    """[(text, facts)] -> [(np.uint16 ids incl. trailing EOT, facts)]."""
    encs = tok.encode_batch([d for d, _ in docs])
    return [(np.asarray(e.ids + [eot], dtype=np.uint16), f) for e, (_, f) in zip(encs, docs)]


# ---------------------------------------------------------------- questions
Q_FORMS = {"city": "Q: Where does {p} live? A:", "job": "Q: What is {p}'s job? A:",
           "pet": "Q: What pet does {p} have? A:", "year": "Q: In what year was {p} born? A:",
           "friend": "Q: Who is {p}'s best friend? A:"}


def make_questions(world, writer, tok, n, seed):
    rng = np.random.default_rng(seed + 1)
    P, E = len(world["people"]), world["early"]
    ppl = world["people"]
    out = []

    def add(qtype, prompt, answer, group, **kw):
        ids = tok.encode(answer).ids
        assert tok.encode(prompt + answer).ids == tok.encode(prompt).ids + ids, (prompt, answer)
        out.append(dict(type=qtype, prompt=prompt, answer=answer, answer_ids=ids, group=group, **kw))

    for r, rel in enumerate(RELS):  # single-hop on the late group, n per relation
        for i in rng.choice(np.arange(E, P), n, replace=False):
            i = int(i)
            add(f"single_{rel}", Q_FORMS[rel].format(p=ppl[i]["name"]), " " + writer.obj(i, rel), "late",
                person=i, fact=i * 5 + r)
    for i in rng.choice(np.arange(E, P), n, replace=False):  # two-hop: country via city
        i = int(i)
        c = world["cities"][ppl[i]["city"]]
        add("two_hop", f"Q: In which country does {ppl[i]['name']} live? A:", " " + world["countries"][c["country"]],
            "late", person=i, fact=i * 5, fact2=5 * P + 2 * c["id"])
    k = 0
    while k < n:  # comparison: who is older (different birth years)
        i, j = (int(v) for v in rng.choice(np.arange(E, P), 2, replace=False))
        if ppl[i]["year"] == ppl[j]["year"]:
            continue
        old = i if ppl[i]["year"] < ppl[j]["year"] else j
        add("compare", f"Q: Who is older, {ppl[i]['name']} or {ppl[j]['name']}? A:", " " + ppl[old]["name"], "late",
            person=i, person2=j, fact=i * 5 + 3, fact2=j * 5 + 3)
        k += 1
    for q, i in enumerate(rng.choice(np.arange(0, E), n, replace=E < n)):  # retention: single-hop on the early group
        i, r = int(i), q % 5
        add("early_retention", Q_FORMS[RELS[r]].format(p=ppl[i]["name"]), " " + writer.obj(i, RELS[r]), "early",
            person=i, fact=i * 5 + r, relation=RELS[r])
    return out


# ---------------------------------------------------------------- cloze items (held-out templates)
def cloze_templates(table, keys, subj):
    """Held-out templates (index >= N_TRAIN_T) usable as cloze: the subject comes before the answer. Returns
    {relation: [(template index, prefix template, answer template)]}; an article before the answer ("{a} {o}") goes
    into the answer, so the prefix never reveals it."""
    out = {}
    for rel in keys:
        for ti in range(N_TRAIN_T, len(table[rel])):
            t = table[rel][ti]
            i = t.index("{o}")
            if subj not in t[:i]:
                continue
            j = i - 4 if t[i - 4:i] == "{a} " else i
            pre, ans = t[:j].rstrip(), " " + t[j:i + 3]
            out.setdefault(rel, []).append((ti, pre, ans))
    return out


def make_cloze(world, tok, questions):
    """Cloze items for the same people / relations as the questions: the prefix of a HELD-OUT template up to the
    answer, e.g. "<Name>'s closest companion is" -> " <friend>". Two-hop has no person -> country template (the
    world never states it), so it is left to the Q: metric; instead geo_country asks the city -> country fact itself
    (held-out geo template). Returns (items, candidate sets): every item names its candidate set and gold index."""
    ppl, P = world["people"], len(world["people"])
    tp = cloze_templates(T_PERSON, RELS, "{p}")
    tg = cloze_templates(T_GEO, ["country"], "{c}")
    jobs = world["jobs"]
    cands = {"city": [" " + c["name"] for c in world["cities"]],
             "job": [" " + j for j in jobs], "job_a": [f" {article(j)} {j}" for j in jobs],
             "pet": [" " + x for x in world["pets"]], "pet_a": [f" {article(x)} {x}" for x in world["pets"]],
             "year": [f" {y}" for y in sorted({p["year"] for p in ppl})],
             "friend": [" " + p["name"] for p in ppl], "country": [" " + c for c in world["countries"]]}
    years = sorted({p["year"] for p in ppl})
    gold_of = {"city": lambda p: p["city"], "job": lambda p: p["job"], "pet": lambda p: p["pet"],
               "year": lambda p: years.index(p["year"]), "friend": lambda p: p["friend"]}
    items = []

    def add(typ, group, pre, ans_t, cset, gold, **kw):
        ans = cands[cset][gold]
        ids = tok.encode(ans).ids
        assert tok.encode(pre + ans).ids == tok.encode(pre).ids + ids, (pre, ans)
        items.append(dict(type=typ, group=group, prefix=pre, answer=ans, answer_ids=ids, cand_set=cset,
                          gold=int(gold), **kw))

    k = 0
    for q in questions:
        t = q["type"]
        if t.startswith("single_") or t == "early_retention":
            rel = t[7:] if t.startswith("single_") else q["relation"]
            p = ppl[q["person"]]
            ti, pre, ans_t = tp[rel][k % len(tp[rel])]
            k += 1
            cset = rel + ("_a" if ans_t.startswith(" {a}") else "")
            typ = f"cloze_{rel}" if t.startswith("single_") else "cloze_early_retention"
            add(typ, q["group"], pre.format(p=p["name"]), ans_t, cset, gold_of[rel](p), person=p["id"],
                relation=rel, template=ti, fact=q["fact"])
    for c in world["cities"]:  # the geography fact itself (the second hop), all 300 cities
        ti, pre, ans_t = tg["country"][c["id"] % len(tg["country"])]
        add("geo_country", "geo", pre.format(c=c["name"]), ans_t, "country", c["country"], city=c["id"], template=ti,
            fact=5 * P + 2 * c["id"])
    return items, cands, {r: [x[0] for x in v] for r, v in list(tp.items()) + [("geo_country", tg["country"])]}


def write_cloze(out, world, tok, questions):
    items, cands, used = make_cloze(world, tok, questions)
    with open(out / "cloze.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    (out / "cloze_candidates.json").write_text(json.dumps(cands))
    print(f"cloze.jsonl: {len(items):,} ({dict(Counter(i['type'] for i in items))}); held-out templates used "
          f"{used}; candidate sets { {k: len(v) for k, v in cands.items()} }", flush=True)
    return items


# ---------------------------------------------------------------- main
class ChunkWriter:
    """Collects documents and writes them in large blocks (many small writes are slow on the X: drive)."""

    def __init__(self, f, block=16_000_000):
        self.f, self.block, self.parts, self.n = f, block, [], 0

    def write(self, ids):
        self.parts.append(ids)
        self.n += len(ids)
        if self.n >= self.block:
            self.flush()

    def flush(self):
        if self.parts:
            np.concatenate(self.parts).tofile(self.f)
            self.parts, self.n = [], 0


def wiki_docs(train, start, chunk=8_000_000, eot=0):
    """Yield Wikipedia documents (np.uint16, incl. the trailing EOT) from train[start:], in order."""
    pos, carry = start, np.zeros(0, np.uint16)
    while pos < len(train):
        buf = np.concatenate([carry, np.asarray(train[pos:pos + chunk])])
        pos += chunk
        ends = np.flatnonzero(buf == eot)
        s = 0
        for e in ends:
            yield buf[s:e + 1]
            s = e + 1
        carry = buf[s:]
    if len(carry):
        yield carry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", default="X:/Datasets/wikipedia_en_20231101_wiki16k")
    ap.add_argument("--out", default="X:/Datasets/wiki_world_mix")
    ap.add_argument("--train-tokens", type=float, default=400e6)
    ap.add_argument("--early-cutoff", type=float, default=0.2 * 200e6,
                    help="early people only before this training-token position (default 20%% of the 12K run)")
    ap.add_argument("--world-frac", type=float, default=0.2)
    ap.add_argument("--geo-frac", type=float, default=0.05, help="share of world documents that are geography docs")
    ap.add_argument("--people", type=int, default=5000)
    ap.add_argument("--early", type=int, default=1000)
    ap.add_argument("--cities", type=int, default=300)
    ap.add_argument("--countries", type=int, default=40)
    ap.add_argument("--rivers", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=200)
    ap.add_argument("--questions", type=int, default=1000, help="per type")
    ap.add_argument("--val-world-frac", type=float, default=0.2, help="world share of the whole val.bin")
    ap.add_argument("--check-tokens", type=float, default=0,
                    help="Wikipedia training tokens scanned for the name check (0 = all the mix uses)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cloze-only", action="store_true",
                    help="only (re)write cloze.jsonl + cloze_candidates.json from --out's world.json / questions.jsonl")
    a = ap.parse_args()
    t0 = time.time()
    wiki, out = Path(a.wiki), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wmeta = json.loads((wiki / "meta.json").read_text())
    assert wmeta["dtype"] == "uint16"
    tok = Tokenizer.from_file(str(wiki / "tokenizer.json"))
    eot = tok.token_to_id(EOT)
    assert eot == wmeta["eot_id"]
    if a.cloze_only:
        world = json.loads((out / "world.json").read_text())
        qs = [json.loads(l) for l in open(out / "questions.jsonl", encoding="utf-8")]
        write_cloze(out, world, tok, qs)
        return
    train = np.memmap(wiki / "train.bin", dtype=np.uint16, mode="r")
    total = int(a.train_tokens)
    need_wiki = int(total * (1 - a.world_frac)) + 100_000
    if need_wiki > len(train):
        raise SystemExit(f"not enough Wikipedia tokens: need ~{need_wiki:,}, have {len(train):,}")
    free = shutil.disk_usage(out).free
    if free < total * 2 * 1.05 + 20e6:
        raise SystemExit(f"disk short: {free/1e9:.1f} GB free at {out}, need ~{total*2/1e9:.1f} GB")

    # ---- world
    n_check = int(a.check_tokens) or need_wiki
    print(f"scanning {n_check/1e6:.0f}M Wikipedia training tokens + val for the name check ...", flush=True)
    wiki_words = wiki_word_set(train, n_check, tok) | wiki_word_set(np.fromfile(wiki / "val.bin", np.uint16), 10**9, tok)
    print(f"  {len(wiki_words):,} distinct words ({time.time()-t0:.0f}s)", flush=True)
    world = build_world(a, tok, wiki_words)
    (out / "world.json").write_text(json.dumps(world, indent=0))
    P, E = a.people, a.early
    n_facts = 5 * P + 2 * a.cities
    ntok = [len(tok.encode(" " + p["name"]).ids) for p in world["people"]]
    print(f"world: {P} people ({E} early), {a.cities} cities, {a.countries} countries, {a.rivers} rivers, "
          f"{len(world['jobs'])} jobs, {len(world['pets'])} pets; {n_facts} facts; tokens per name "
          f"{np.mean(ntok):.2f} (min {min(ntok)}, max {max(ntok)})", flush=True)

    # ---- training stream
    train_w = Writer(world, list(range(N_TRAIN_T)))
    rng = np.random.default_rng(a.seed + 2)
    pool_all, pool_late = np.arange(P), np.arange(E, P)
    cutoff = int(a.early_cutoff)
    marks = sorted(m for m in {cutoff, int(200e6), total} if 0 < m <= total)
    exp = np.zeros(n_facts, np.int64)        # exposures so far
    exp_at = {}                              # snapshots at the marks
    early_phase = cutoff > 0
    queue = []
    pos = w_tok = k_tok = n_wdocs = n_kdocs = 0
    next_print, samples = 0, []
    ratio = a.world_frac / (1 - a.world_frac)

    def refill():
        pool = pool_all if early_phase else pool_late
        docs = [train_w.doc(rng, pool, a.geo_frac) for _ in range(4000)]
        if len(samples) < 40:
            samples.extend(d for d, _ in docs[:40])
        return encode_docs(tok, docs, eot)

    with open(out / "train.bin", "wb") as raw:
        f = ChunkWriter(raw)
        docs = wiki_docs(train, 0, eot=eot)
        while pos < total:
            # one Wikipedia article, then world documents until the world share is back at world_frac
            d = next(docs)[:total - pos]
            f.write(d)
            pos += len(d)
            k_tok += len(d)
            n_kdocs += 1
            while pos < total and w_tok < ratio * k_tok:
                if early_phase and pos >= cutoff:
                    early_phase, queue = False, []   # drop documents generated with early people
                if not queue:
                    queue = refill()
                ids, facts = queue.pop()
                if early_phase and pos + len(ids) > cutoff:  # would straddle the cutoff: switch now
                    early_phase, queue = False, []
                    continue
                ids = ids[:total - pos]
                for m in marks:  # snapshot the exposures when the stream crosses a mark
                    if pos < m <= pos + len(ids) and m not in exp_at:
                        exp_at[m] = exp.copy()
                f.write(ids)
                np.add.at(exp, facts, 1)
                pos += len(ids)
                w_tok += len(ids)
                n_wdocs += 1
            for m in marks:
                if pos >= m and m not in exp_at:
                    exp_at[m] = exp.copy()
            if pos >= next_print:
                el = time.time() - t0
                print(f"  train: {pos/1e6:.1f}M / {total/1e6:.0f}M tokens (world {w_tok/max(pos,1):.3f}, "
                      f"{n_wdocs:,} world docs, {n_kdocs:,} wiki docs, {'early+late' if early_phase else 'late'} "
                      f"people) {el:.0f}s", flush=True)
                next_print += 10_000_000
        f.flush()
    for m in marks:
        exp_at.setdefault(m, exp.copy())
    print(f"train.bin: {pos:,} tokens ({w_tok:,} world = {w_tok/pos:.3f}, {k_tok:,} Wikipedia), "
          f"{n_wdocs:,} world docs, {n_kdocs:,} wiki docs ({time.time()-t0:.0f}s)", flush=True)

    # ---- exposure statistics
    def ex_stats(e):
        pe = e[:5 * P].reshape(P, 5)
        g = lambda x: dict(mean=float(x.mean()), median=float(np.median(x)), min=int(x.min()), max=int(x.max()))
        return dict(early=g(pe[:E]), late=g(pe[E:]), geo=g(e[5 * P:]),
                    early_by_rel={r: float(pe[:E, k].mean()) for k, r in enumerate(RELS)},
                    late_by_rel={r: float(pe[E:, k].mean()) for k, r in enumerate(RELS)})
    exposure = {f"{m/1e6:g}M": ex_stats(exp_at[m]) for m in sorted(exp_at)}
    np.savez(out / "exposures.npz", **{f"at_{m}": exp_at[m] for m in sorted(exp_at)}, final=exp)
    for k, v in exposure.items():
        print(f"  exposures per fact at {k}: early mean {v['early']['mean']:.1f} (min {v['early']['min']}), late mean "
              f"{v['late']['mean']:.1f} (min {v['late']['min']}), geo mean {v['geo']['mean']:.1f}", flush=True)

    # ---- validation: Wikipedia val + fresh world documents with the 2 held-out templates
    val_w = Writer(world, list(range(N_TRAIN_T, 12)))
    vrng = np.random.default_rng(a.seed + 3)
    wval = np.fromfile(wiki / "val.bin", dtype=np.uint16)
    target = int(len(wval) * a.val_world_frac / (1 - a.val_world_frac))
    vparts, vn, vdocs, vsamples = [], 0, 0, []
    while vn < target:
        docs = [val_w.doc(vrng, pool_all, a.geo_frac) for _ in range(2000)]
        if not vsamples:
            vsamples = [d for d, _ in docs[:5]]
        for ids, _ in encode_docs(tok, docs, eot):
            vparts.append(ids)
            vn += len(ids)
            vdocs += 1
            if vn >= target:
                break
    vworld = np.concatenate(vparts)
    vworld.tofile(out / "val_world.bin")
    np.concatenate([wval, vworld]).tofile(out / "val.bin")
    shutil.copy(wiki / "tokenizer.json", out / "tokenizer.json")
    print(f"val.bin: {len(wval):,} Wikipedia + {len(vworld):,} world tokens ({vdocs:,} docs)", flush=True)

    # ---- questions
    qs = make_questions(world, train_w, tok, a.questions, a.seed)
    with open(out / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q) + "\n")
    # held-out check: no question form appears in the training templates
    all_t = " ".join(t for ts in list(T_PERSON.values()) + list(T_GEO.values()) for t in ts)
    assert "Q:" not in all_t and "A:" not in all_t and "?" not in all_t
    print(f"questions.jsonl: {len(qs):,} ({dict(Counter(q['type'] for q in qs))})", flush=True)
    write_cloze(out, world, tok, qs)

    with open(out / "samples.txt", "w", encoding="utf-8") as f:
        f.write("=== training world documents ===\n\n" + "\n\n---\n\n".join(samples[:20]))
        f.write("\n\n=== val world documents (held-out templates) ===\n\n" + "\n\n---\n\n".join(vsamples[:5]))
    meta = {k: v for k, v in wmeta.items() if k not in ("train_pos",)}
    meta.update(
        source=f"wiki_world_mix: {wmeta['source']} (train.bin in order) + generated world (generate_world.py)",
        train_tokens=int(pos), train_world_tokens=int(w_tok), train_wiki_tokens=int(k_tok),
        train_world_docs=int(n_wdocs), train_wiki_docs=int(n_kdocs), world_frac=a.world_frac,
        train_articles=int(n_kdocs), train_files=f"Wikipedia: the first {k_tok:,} tokens of {wiki / 'train.bin'}",
        val_tokens=int(len(wval) + len(vworld)), val_wiki_tokens=int(len(wval)), val_world_tokens=int(len(vworld)),
        val_note="val.bin = Wikipedia val.bin, then val_world.bin (fresh world docs, held-out templates 10-11)",
        early_cutoff=cutoff, early_people=E, world=dict(
            seed=a.seed, people=P, early=E, cities=a.cities, countries=a.countries, rivers=a.rivers,
            jobs=len(world["jobs"]), pets=len(world["pets"]), facts=n_facts, templates_per_relation=12,
            train_templates=N_TRAIN_T, geo_frac=a.geo_frac, questions_per_type=a.questions,
            question_types=sorted(Counter(q["type"] for q in qs))),
        exposures=exposure, done=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"done in {time.time()-t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
