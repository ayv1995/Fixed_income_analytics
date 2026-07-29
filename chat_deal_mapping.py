import re
from dateutil.parser import parse as parse_date
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    lit, col, when, udf, count, sum as spark_sum,
    to_date, abs as spark_abs, datediff, lower, trim, regexp_replace, row_number
)
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.window import Window

INPUT_FILE_PATH = "/Volumes/mbtmptt_applications_syst/playground/raw_files/transcript_castss.txt"

# --- Regex patterns ---
TIME_RE = re.compile(r"^(\d{2}:\d{2}:\d{2})\s+(.*)$")
DATE_RE = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(.+)$")
PRICING_RE = re.compile(r"(€€€.*PRICED|\bpriced\b|BOOKS CLOSED)", re.IGNORECASE)

ISSUER_PATTERNS = [
    re.compile(r"NEW MANDATE:\s*(.+?)(?:\s+(?:Inaugural|EUR|\d+[.-]?(?:yr|year|YR)|Senior|Fixed|Floating|Bond|Green)|\s*[–\-]|\s*€€€)"),
    re.compile(r"^([^,(]+)[,(].*has mandated", re.IGNORECASE),
    re.compile(r"^([^(]+)\(Ticker:"),
]

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


def extract_issuer(line):
    """Try each issuer pattern, return first match or None."""
    for pat in ISSUER_PATTERNS:
        m = pat.search(line)
        if m:
            return m.group(1).strip()
    return None


# --- Parse transcript ---
with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

rows = []
current_date = None
current_phase = "PRE_ROADSHOW"
deal_date = None
issuer = "UNKNOWN"
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

LEGAL_SUFFIXES = r"\s*(plc|ltd|inc|a/s|ab|as|gmbh|sa|nv|se|financial services|publ|group)\s*"
def clean_issuer_col(c):
    return trim(regexp_replace(lower(c), LEGAL_SUFFIXES, " "))

df_join = (
    df.withColumn("deal_date_parsed", to_date(col("deal_date").substr(1, 10)))
      .withColumn("issuer_clean", clean_issuer_col(col("issuer_name")))
      .withColumn("_row_id", row_number().over(Window.orderBy("comment_date", "comment_time")))
)

df2_join = (
    df2.withColumn("creation_date", to_date(col("CreationDateTimeStamp")))
       .withColumn("issuer_clean2", clean_issuer_col(col("Issuer_Name")))
)

# Fuzzy join: cleaned name contains + date within 7 days
joined = df_join.join(
    df2_join,
    (df2_join.issuer_clean2.contains(df_join.issuer_clean) |
     df_join.issuer_clean.contains(df2_join.issuer_clean2)) &
    (spark_abs(datediff(df_join.deal_date_parsed, df2_join.creation_date)) <= 7),
    "left"
)

# Deduplicate: pick closest date match per row to keep row count = df count
w = Window.partitionBy("_row_id").orderBy(spark_abs(datediff(df_join.deal_date_parsed, df2_join.creation_date)))
joined = joined.withColumn("_rank", row_number().over(w)).filter(col("_rank") == 1).drop("_rank", "_row_id")

# Clean up columns
joined = (
    joined.drop("deal_date_parsed", "creation_date", "issuer_clean", "issuer_clean2")
          .drop(df_join.issuer_name)
          .drop("deal_date")
          .withColumnRenamed("Issuer_Name", "issuer_name")
          .withColumnRenamed("CreationDateTimeStamp", "creation_date")
)

# Final output
joined.select(
    "DbKey", "Name", "issuer_name", "IssuerCountry", "creation_date",
    "comment_date", "comment_time", "comments", "phase",
    "commenter_name", "commenter_company"
).show(20, truncate=80)

print(f"df rows: {df.count()} | joined rows: {joined.count()}")
