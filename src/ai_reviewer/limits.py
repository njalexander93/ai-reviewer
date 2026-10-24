"""Central finite work policy shared by collection, planning, contracts and adapters.

Byte limits measure UTF-8 serialized evidence unless named for raw blobs. Token
limits use provider counts. Derived finding/group bounds follow attempts and work;
changing an upstream maximum requires checking downstream payload/cost limits.
These are rejection thresholds, not promises that every maximum-sized run fits.
"""

# Source and packet sizes: collection may exceed one provider input.
PACKET_BYTES = 600_000
UNIT_BYTES = 160_000
CHUNK_BYTES = 420_000
CAPACITY_PACKET_BYTES = 200_000
CONSOLIDATION_BYTES = 200_000
MAX_CHUNKS = 64
MAX_SNAPSHOT_BYTES = 32_000_000
MAX_BLOB_BYTES = 8_000_000
MAX_WORK = 128
MAX_RELATIONS = 20_000
MAX_FILES = 3_000
MAX_INPUT_TOKENS = 250_000
MODEL_REPORT_BYTES = 60_000
GITHUB_BODY_BYTES = 60_000
PROVIDER_TIMEOUT_SECONDS = 300

# Work fan-out and schema bounds keep all original findings representable.
PROVIDERS = 2
MAX_ATTEMPTS = 3
MAX_FINDINGS = 40
MAX_SOURCE_REQUESTS = 8
MAX_COMPLETED_FINDINGS = MAX_ATTEMPTS * MAX_FINDINGS
MAX_FINAL_GROUPS = MAX_WORK * PROVIDERS * MAX_COMPLETED_FINDINGS
MAX_GROUP_SOURCES = PROVIDERS * MAX_FINDINGS
MAX_CONSOLIDATION_GROUPS = PROVIDERS * MAX_FINDINGS
MAX_CONSOLIDATION_CALLS = 128
MAX_SUMMARY_CALLS = 128
SUMMARY_PAGE_SIZE = 8
MAX_OUTPUT_TOKENS = 16_000
# Publication and collection caps remain independent of model work limits.
MAX_GITHUB_COMMENTS = 1_000
MAX_GITHUB_PAYLOAD_BYTES = 8_000_000
MAX_GITHUB_REQUESTS = 10_000
