from pathlib import Path

# Compute project root relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up two levels from script to reach project root

# Define directory paths
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Create directories if they don't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define paths to the pre-trained BioFM model and tokenizer
MODEL_PATH = "m42-health/BioFM-265M"
TOKENIZER_PATH = "m42-health/BioFM-265M"

# Direct URL constants for data files
# Gene annotation
DEFAULT_GENE_ANNOTATION_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/gencode.v48.annotation.gff3.gz"

# Reference genome
DEFAULT_REFERENCE_GENOME_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_48/GRCh38.primary_assembly.genome.fa.gz"

# VCF example
DEFAULT_VCF_URL = "https://raw.githubusercontent.com/dna-seq/prs/339f05b6aa68099c31bc77fc5abf28e24bd782a3/data/example.vcf" 