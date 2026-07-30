import re
from dateutil.parser import parse as parse_date
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    lit, col, when, udf, count, sum as spark_sum,
    to_date, abs as spark_abs, datediff, lower, trim, regexp_replace, row_number,
    split, array_intersect, size, least
)
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.window import Window

INPUT_FILE_PATH = "/Volumes/mbtmptt_applications_syst/playground/raw_files/transcript_genmab_passive.txt"

# --- Regex patterns ---
TIME_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})\s+(.*)$")
DATE_RE = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(.+)$")
PRICING_RE = re.compile(r"(€€€.*PRICED|\bpriced\b|BOOKS CLOSED)", re.IGNORECASE)

# Simplified: find "has mandated" (issuer is subject) or "Issuer" label
_HAS_MANDATED_RE = re.compile(r'(.+?)\s+has\s+mandated', re.IGNORECASE)
_ISSUER_LABEL_RE = re.compile(r'^Issuer[:\s]*([A-Z].+)', re.IGNORECASE)

TICKER_RE = re.compile(r"\(Ticker:\s*([A-Z0-9]+)\)")

PHASE_PATTERNS = [
    ("POST_PRICING", re.compile(r"(€€€.*PRICED|\bpriced\b)", re.IGNORECASE)),
    ("AT_FINALS",    re.compile(r"(FINAL TERMS|FINALS|\bfinals\b)", re.IGNORECASE)),
    ("AT_GUIDANCE",  re.compile(r"(GUIDANCE|REVISED|REVISED IPT|NARROW)", re.IGNORECASE)),
    ("BOOK_BUILD",   re.compile(r"(BOOKS OPEN|BOOK.?BUILD|\bIPT\b|INITIAL PRICE)", re.IGNORECASE)),
    ("ROADSHOW",     re.compile(r"(ROADSHOW|INVESTOR (CALL|MEETING)|CALL SCHEDULE)", re.IGNORECASE)),
]


def extract_commenter(msg, name_map):
    """Extract commenter name from message. Only accept if in invite map."""
    patterns = [
        re.match(r"^([A-Z][A-Z\s'\-]{3,}):", msg),
        re.match(r"\*\*\*\s+([A-Z][A-Z\s'\-]+?)\s*\(", msg),
        re.search(r"\s([A-Z][A-Z\s]{3,})$", msg),
    ]
    for m in patterns:
        if m:
            name = m.group(1).strip()
            if name in name_map:
                return name
            break
    return ""


# Words that indicate a bond description, not an issuer name
_NOT_ISSUER_RE = re.compile(
    r'^(EUR|USD|GBP|NOK|SEK|DKK|CHF|JPY|AUD|CAD)\b|\b(Benchmark|FRN|Fixed|Floating|Senior|Subordinated|Covered|Green|Notes|Bond)\b',
    re.IGNORECASE
)

def _clean_issuer_result(text):
    """Take only company name: strip ratings, tickers, LEI codes."""
    text = re.split(r'[,(]', text)[0].strip()
    text = re.sub(r'[A-Z0-9]{20,}', '', text).strip()
    text = re.sub(r'^(?:Issuer|Emittent)[:\s]*', '', text, flags=re.IGNORECASE).strip()
    return text

# Strategy 3: passive transcripts - issuer mentioned in deal context
_PASSIVE_ISSUER_RE = re.compile(
    r'(?:IPT|pricing|spread|deal|transaction|mandate|books?)\s+(?:for|on|of|from)\s+([A-Z][A-Za-z][A-Za-z0-9 .&-]{1,40}?)(?:\s*[?!.,;]|\s+(?:is|are|was|will|has|today|tomorrow|at|@))',
    re.IGNORECASE
)

def extract_issuer(line):
    """Extract issuer - three strategies for active and passive transcripts."""
    # Strategy 1: "X has mandated" - issuer is always the subject
    m = _HAS_MANDATED_RE.search(line)
    if m:
        result = _clean_issuer_result(m.group(1))
        if result and not _NOT_ISSUER_RE.search(result):
            return result

    # Strategy 2: "Issuer: X" or "IssuerX" label
    m = _ISSUER_LABEL_RE.match(line)
    if m:
        result = _clean_issuer_result(m.group(1))
        if result and not _NOT_ISSUER_RE.search(result):
            return result

    # Strategy 3: passive - "IPT for X?", "deal from X", etc.
    m = _PASSIVE_ISSUER_RE.search(line)
    if m:
        result = m.group(1).strip()
        if result and len(result) > 2 and not _NOT_ISSUER_RE.search(result):
            return result

    return None


# --- Parse transcript ---
with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

rows = []
current_date = None
current_phase = "PRE_ROADSHOW"
deal_date = None
issuer = "UNKNOWN"
ticker = None
name_company_map = {}

for line in lines:
    line = line.strip()
    if not line:
        continue

    if DATE_RE.match(line):
        current_date = parse_date(line).strftime("%Y-%m-%d")
        continue

    if not name_company_map and "invited" in line:
        for m in re.finditer(r"([A-Z][A-Z\s'\-]+?)\s*\(([^)]+)\)", line):
            name_company_map[m.group(1).strip()] = m.group(2).strip()

    for phase_name, phase_re in PHASE_PATTERNS:
        if phase_re.search(line):
            current_phase = phase_name
            break

    time_match = TIME_RE.match(line)
    if time_match:
        msg = time_match.group(2)
        commenter = extract_commenter(msg, name_company_map)
        rows.append([current_date, time_match.group(1), msg, current_phase, commenter])
    elif rows:
        rows[-1][2] += " " + line

    if deal_date is None and PRICING_RE.search(line):
        t = time_match.group(1) if time_match else ""
        deal_date = f"{current_date} {t}".strip()

    if issuer == "UNKNOWN":
        found = extract_issuer(line)
        if found:
            issuer = found

# Fallback: if no pricing line found, use first date in transcript
if deal_date is None and current_date:
    deal_date = current_date

# Forward-fill commenter
for i in range(1, len(rows)):
    if not rows[i][4]:
        rows[i][4] = rows[i-1][4]

print(f"Issuer: {issuer} | Deal date: {deal_date} | Rows: {len(rows)}")

# --- Build DataFrame ---
spark = SparkSession.builder.getOrCreate()

schema = StructType([
    StructField("comment_date", StringType(), True),
    StructField("comment_time", StringType(), True),
    StructField("comments", StringType(), True),
    StructField("phase", StringType(), True),
    StructField("commenter_name", StringType(), True),
])
df = spark.createDataFrame(rows, schema)

get_company_udf = udf(lambda name: name_company_map.get(name) if name else None, StringType())
df = (
    df.withColumn("commenter_company", get_company_udf(col("commenter_name")))
      .withColumn("issuer_name", lit(issuer))
      .withColumn("deal_date", lit(deal_date))
)
df.show(truncate=100)

# Distinct comment dates
df.select("comment_date").distinct().orderBy("comment_date").show()

# --- Join with deals table (keep same row count as df) ---
df2 = spark.table("mbtmptt_applications_syst.playground.primary_market_data_deals")

def clean_issuer_col(c):
    """Clean issuer name for fuzzy matching — generic, not transcript-specific."""
    cleaned = lower(c)
    # Remove "issuer:" or "emittent:" prefix
    cleaned = regexp_replace(cleaned, '^(?:issuer|emittent)[: ]+', '')
    # Normalize Nordic/special characters
    cleaned = regexp_replace(cleaned, 'ø', 'o')   # ø → o
    cleaned = regexp_replace(cleaned, 'æ', 'ae')  # æ → ae
    cleaned = regexp_replace(cleaned, 'å', 'a')   # å → a
    cleaned = regexp_replace(cleaned, 'ö', 'o')   # ö → o
    cleaned = regexp_replace(cleaned, 'ä', 'a')   # ä → a
    cleaned = regexp_replace(cleaned, 'ü', 'u')   # ü → u
    cleaned = regexp_replace(cleaned, 'é', 'e')   # é → e
    # Remove parenthetical content like (Austria), (Ticker: ...)
    cleaned = regexp_replace(cleaned, '\\([^)]*\\)', ' ')
    # Remove dots (P.S.K. -> PSK)
    cleaned = regexp_replace(cleaned, '\\.', '')
    # Remove hyphens, dashes, and trailing punctuation
    cleaned = regexp_replace(cleaned, '[-\\x{2013}\\x{2014}]', ' ')
    cleaned = regexp_replace(cleaned, '[,;:]+', ' ')
    # Remove common legal suffixes (word boundaries prevent matching inside words)
    cleaned = regexp_replace(cleaned, '(?i)\\b(plc|ltd|limited|inc|corp|ag|gmbh|sa|nv|se|ab|as|a/s|publ|group|bank|financial services|unlimited company)\\b', ' ')
    # Collapse whitespace
    return trim(regexp_replace(cleaned, '\\s+', ' '))

df_join = (
    df.withColumnRenamed("issuer_name", "_transcript_issuer")
      .withColumn("deal_date_parsed", to_date(col("deal_date").substr(1, 10)))
      .withColumn("issuer_clean", clean_issuer_col(col("_transcript_issuer")))
      .withColumn("_row_id", row_number().over(Window.orderBy("comment_date", "comment_time")))
)

df2_join = (
    df2.withColumn("creation_date", to_date(col("CreationDateTimeStamp")))
       .withColumn("issuer_clean2", clean_issuer_col(col("Issuer_Name")))
)

# Fuzzy join: word overlap >= 2/3 of shorter name + date within 7 days
df_join = df_join.withColumn("_words1", split(col("issuer_clean"), " "))
df2_join = df2_join.withColumn("_words2", split(col("issuer_clean2"), " "))

overlap = size(array_intersect(col("_words1"), col("_words2")))
min_len = least(size(col("_words1")), size(col("_words2")))

# Join: name overlap >= 2/3 of shorter name + date within 7 days
joined = df_join.join(
    df2_join,
    (overlap >= (min_len * 2 / 3)) &
    (spark_abs(datediff(df_join.deal_date_parsed, df2_join.creation_date)) <= 7),
    "left"
)

# Deduplicate: pick closest date match
w = Window.partitionBy("_row_id").orderBy(spark_abs(datediff(df_join.deal_date_parsed, df2_join.creation_date)))
joined = joined.withColumn("_rank", row_number().over(w)).filter(col("_rank") == 1).drop("_rank", "_row_id")

# Clean up columns
joined = joined.drop("deal_date_parsed", "creation_date", "issuer_clean", "issuer_clean2", "_words1", "_words2", "deal_date")
joined = joined.drop("_transcript_issuer")
joined = joined.withColumnRenamed("Issuer_Name", "issuer_name")
joined = joined.withColumnRenamed("CreationDateTimeStamp", "creation_date")

# Final output
joined.select(
    "DbKey", "Name", "issuer_name", "IssuerCountry", "creation_date",
    "comment_date", "comment_time", "comments", "phase",
    "commenter_name", "commenter_company"
).show(20, truncate=80)

print(f"df rows: {df.count()} | joined rows: {joined.count()}")
